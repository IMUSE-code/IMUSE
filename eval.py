import argparse
import os
from pathlib import Path
import json
import numpy as np

import torch
from torch.cuda.amp import GradScaler
from torch.utils import data


from dataset import IMUArkitDataset
from network.transformer_cond_diffusion_model import CondGaussianDiffusion

from ict_model.face_model_io import convert_ict_to_arkit
from types import SimpleNamespace

from tqdm import tqdm
import pandas as pd

class Tester(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        amp = False,
    ):
        super().__init__()

        self.model = diffusion_model

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.opt = opt 

        self.prep_dataloader()

        self.window = opt.window 

    def prep_dataloader(self):
        test_dataset = IMUArkitDataset(            
            self.opt.dataset_datapath_list, 
            window_size = self.opt.window, 
            imu_desired_cols = self.opt.dataset_imu_desired_cols,
            arkit_to_ict_file = self.opt.dataset_arkit_to_ict_file,
            mode='test', 
            overlap=self.opt.dataset_overlap,
        )
        self.ds = test_dataset
        self.dl = data.DataLoader(self.ds, batch_size=self.opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        
    def load_weight_path(self, weight_path):
        data = torch.load(weight_path)

        self.model.load_state_dict(data['model'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def test(self):
        self.load_weight_path(self.opt.weight_path)

        output_data = {}
        for data_dict in tqdm(self.dl):
            imu = data_dict['imu'].cuda()
            bs = data_dict['bs'].cuda()

            padding_mask = self.prep_padding_mask(torch.zeros_like(bs), data_dict['seq_len'])

            bs_pred = self.model.sample(torch.zeros_like(bs), imu, padding_mask=padding_mask)

            for i, in_datapath in enumerate(data_dict['datapath']):
                if in_datapath not in output_data.keys():
                    output_data[in_datapath] = {
                        "save_npy": np.empty((0, opt.repr_dim)),
                        "gt_npy": np.empty((0, opt.repr_dim)),
                        }
                
                start_time_idx = int(torch.where(data_dict['idxs'][i] == len(output_data[in_datapath]["save_npy"]))[0])

                bs_pred_save = bs_pred[i].detach().cpu().numpy()
                bs_gt_save = bs[i].detach().cpu().numpy()
                output_data[in_datapath]["save_npy"] = np.concatenate([output_data[in_datapath]["save_npy"], bs_pred_save[start_time_idx:, :]], axis=0)
                output_data[in_datapath]["gt_npy"] = np.concatenate([output_data[in_datapath]["gt_npy"], bs_gt_save[start_time_idx:, :]], axis=0)


        self.output_data = output_data
        print('testing complete')

    def eval(self):

        output_data = self.output_data
        save_dir = Path(opt.output_dir)

        eval_result = {}
        testset_list = []
        for in_datapath in output_data.keys():
            in_datapath_pathlib = Path(in_datapath)
            out_datapath = os.path.join(save_dir, in_datapath_pathlib.parts[-1])
            os.makedirs(out_datapath, exist_ok=True)

            l1_loss = np.abs(output_data[in_datapath]["save_npy"] - output_data[in_datapath]["gt_npy"])
            mean_l1_loss = np.mean(l1_loss)

            l2_loss = (output_data[in_datapath]["save_npy"] - output_data[in_datapath]["gt_npy"]) ** 2
            mean_l2_loss = np.mean(l2_loss)

            eval_result[in_datapath] = {
                "l1_loss": mean_l1_loss, 
                "l2_loss": mean_l2_loss, 
            }

            out_ict_file = os.path.join(out_datapath, 'pred_ict_params.npy')
            np.save(out_ict_file, output_data[in_datapath]["save_npy"])

            arkit_data = convert_ict_to_arkit(output_data[in_datapath]["save_npy"])
            out_arkit_file = os.path.join(out_datapath, 'pred_arkit_params.csv')
            arkit_data.to_csv(out_arkit_file, index=False)

            testset_list.append(out_arkit_file)

        with open(save_dir / "eval_result.json", 'w') as f:
            json.dump(eval_result, f, indent=4)
        
        with open(save_dir / "testset_list.json", 'w') as f:
            json.dump(testset_list, f, indent=4)
        
        with open(save_dir / "visualize.json", 'w') as f:
            json.dump({
                "input": testset_list,
                "identity": "ict_model/models/sample_identity_coeffs.json",
                "arkit_to_ict_file": "configs/arkit_to_ict.json",
                "base_model": "ict_model/models/FaceXModelTriangle",
                "fps": 60
            }, f, indent=4)
            
        print('save complete')
    
    def prep_padding_mask(self, val_data, seq_len):
        # Generate padding mask 
        actual_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device)

        return padding_mask 

def run_test(opt, device):
    # Define model  
    imu = opt.n_imu * len(opt.dataset_imu_desired_cols)

    loss_type = opt.loss_type
    
    diffusion_model = CondGaussianDiffusion(d_feats=opt.repr_dim, d_model=opt.d_model, d_imu=imu, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=opt.repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type)
  
    diffusion_model.to(device)

    tester = Tester(
        opt,
        diffusion_model,
        amp=True,                        # turn on mixed precision
    )

    tester.test()
    tester.eval()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', '-d', default='./data/test_config.json', type=str)
    parser.add_argument('--output_dir', '-o', default='./output', type=str)
    parser.add_argument('--weight_path', '-m', default='./checkpoints/IMUSE_model_best.pt', type=str)
    parser.add_argument('--device', default='0', type=str)
    
    parser.add_argument('--window', default=120, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_dec_layers', default=4, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--d_k', default=256, type=int)
    parser.add_argument('--d_v', default=256, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--n_imu', default=11, type=int)
    parser.add_argument('--repr_dim', default=53, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)

    opt = parser.parse_args()
    
    with open(opt.dataset_config, 'r') as file:
        opt = opt.__dict__
        opt.update(json.load(file))

    opt = SimpleNamespace(**opt)

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    run_test(opt, device)