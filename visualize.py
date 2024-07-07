import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm
import open3d as o3d
import subprocess
import json
import pandas as pd
from ict_model import face_model_io
from types import SimpleNamespace


def bs_to_obj(data, output_folder, id_coeffs, face_model):
    """Reads an ICT FaceModel .json file and writes its mesh.
    """
    # Create a new FaceModel and load the model
    for i in tqdm(range(len(data))):
        ex_coeffs = data[i]
        face_model.from_coefficients(id_coeffs, ex_coeffs)
        face_model.deform_mesh()
        output_filename = output_folder + '/obj/{}.obj'.format(str(i+1).zfill(8)) 
        face_model_io.write_deformed_mesh(output_filename, face_model)

def obj_to_png(output_folder, width, height):
    renderer = o3d.visualization.Visualizer()
    renderer.create_window(width=width, height=height)
    for i in tqdm(range(0,len(os.listdir(output_folder + '/obj')))):
        output_filename = output_folder + '/obj/{}.obj'.format(str(i+1).zfill(8))
        model = o3d.io.read_triangle_mesh(output_filename)
        model.compute_vertex_normals()

        normals = np.asarray(model.vertex_normals)
        colors = (normals * 0.5 + 0.5)
        
        grayscale_colors = np.mean(colors, axis=-1)
        grayscale_image = np.stack([grayscale_colors] * 3, axis=-1)

        model.vertex_colors = o3d.utility.Vector3dVector(grayscale_image)

        renderer.add_geometry(model)

        output_file_path = output_filename.replace('obj', 'png')

        renderer.capture_screen_image(output_file_path, do_render=True)
        renderer.clear_geometries()
    renderer.destroy_window()
    return

def png_to_vid(input_folder, output_video, width=800, height=600, framerate=30, codec='libx264'):
    cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', os.path.join(input_folder, '%08d.png'),
        '-c:v', codec,
        '-s', f'{width}x{height}',
        '-pix_fmt', 'yuv420p',
        '-y',
        output_video
    ]

    subprocess.run(cmd)
    print(output_video)
    return


def main(args):
    id_coeffs, _ = face_model_io.read_coefficients(args.identity)
    id_coeffs *= 0
    face_model = face_model_io.load_face_model(args.base_model)

    codec='libx264'
    width=1080
    height=1440
        
    for input_folder in args.input:
        # Visualize file
        args.output = os.path.dirname(input_folder)
        os.makedirs(args.output, exist_ok=True)
        
        data = pd.read_csv(input_folder)
        with open(args.arkit_to_ict_file, 'r') as f:
            llf_mapping = pd.read_json(f)
        desired_columns = llf_mapping[0].tolist()
        data = data[desired_columns].to_numpy()
        
        os.makedirs(args.output + '/obj', exist_ok=True)
        bs_to_obj(data, args.output, id_coeffs, face_model)
        
        os.makedirs(args.output + '/png', exist_ok=True)
        obj_to_png(args.output, width, height)
        
        png_to_vid(args.output + '/png', os.path.join(args.output, "arkit_vis.mp4"), width, height, args.fps, codec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="output/visualize.json", type=str)
    opt = parser.parse_args()
    with open(opt.config, 'r') as file:
        config = json.load(file)
    
    opt = SimpleNamespace(**config)
    
    main(opt)