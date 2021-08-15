import os 
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import cc3d 
import yaml

from tqdm import tqdm
from collections import defaultdict

from utils.preprocess import Crop, Resize
from utils.stitcher import Stitcher2d
from utils.parser import Parser
from utils.utils import get_time, _padding, _patch_offset_generation, _sym_padding


if __name__ == '__main__':
    # parse argumets and set parameters
    args = Parser().get_args()
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for k, v in config.items():
            args.__setattr__(k, v)

    device = torch.device("cuda")
    suborgan_net = torch.jit.load(args.model['suborgan']).to(device)
    cell_by_cell_net = torch.jit.load(args.model['cell_by_cell']).to(device)

    file = args.data['path']
    times = get_time(file)
    for time in tqdm(times):
        with h5py.File(file, 'r') as f:
            try:
                z_resolution = f['Data/3D'].attrs['ResolutionZ'][0]
            except:
                z_resolution = f['Data/3D'].attrs['ResolutionZ']
            ri = f['Data/3D/' + time][...]
            o_ri = ri
            if ri.dtype == 'uint16':
                ri = (ri.clip(13370, 13900) - 13370) / (13900 - 13370)
            else:
                ri = (ri.clip(1.337, 1.390) - 1.337) / (1.390 - 1.337)
            ri_np = ri.astype(np.float32)
        
        resize = Resize((args.preprocess['z_size'], args.preprocess['zoomed_size'], args.preprocess['zoomed_size']), order=0)
        center_z_crop = Crop((args.preprocess['center_crop_size'], ri_np.shape[1], ri_np.shape[2]), center=(True, False, False))

        ri = _padding(ri_np, (1, 1))
        ri = np.stack([ri])
        for preprocess in [center_z_crop, resize]:
            ri = preprocess(ri)
        ri = ri[0]
        ri_pad = _sym_padding(args.preprocess['patch_size'], ri)
        patch_offsets = _patch_offset_generation(args.preprocess['patch_size'], ri_pad)
        ri_pad = torch.from_numpy(ri_pad)[None, None].to(device)

        patches = defaultdict(dict)
        stitcher = Stitcher2d(args.preprocess['z_size'], args.preprocess['patch_size'], ri.shape[1:])
        with torch.no_grad():
            for offset in tqdm(patch_offsets):
                patch_img = ri_pad[:, :, :, offset[0]:offset[0] + args.preprocess['patch_size'], offset[1]:offset[1] + args.preprocess['patch_size']]
                o1 = torch.sigmoid(suborgan_net(patch_img))
                img_dict = {'ri': patch_img.cpu().numpy()[0][0]}
                for j, organ in enumerate(["nucleus", "nucleoli", "membrane", "lipid"]):
                    img_dict[organ] = o1[0][j].cpu().numpy()
                patches[offset] = img_dict
            img = stitcher.stitch(patches)

        #cell by cell
        cell_shape = (args.preprocess['z_size'], args.preprocess['cell_resize_size'], args.preprocess['cell_resize_size'])
        ri_tensor = F.interpolate(torch.from_numpy(ri)[None, None].to(device), size=cell_shape, mode='nearest')
        nuc = F.interpolate(torch.tensor(img['nucleus'] / 4).view(1, 1, *img['nucleus'].shape).float(), size=cell_shape, mode='nearest').to(device)
        mem = F.interpolate(torch.tensor(img['membrane'] / 4).view(1, 1, *img['membrane'].shape).float(), size=cell_shape, mode='nearest').to(device)

        zero = torch.tensor(0, dtype=torch.float).to(device)
        
        seed_threshold = nuc[0, 0].round_().cpu().int().numpy()
        seed_label = cc3d.connected_components(seed_threshold, out_dtype=np.uint16) # 26-connected
        seed_label = torch.from_numpy(seed_label.astype(float)).to(device)[None, None]
        
        num_features = int(seed_label.max())
        out_inst = torch.zeros((num_features, args.preprocess['center_crop_size'], args.preprocess['zoomed_size'], args.preprocess['zoomed_size']), dtype=torch.float, device=ri_tensor.device)

        empty_inst = torch.empty((1, 4, *ri_tensor.shape[2:]), device=ri_tensor.device)
        empty_inst[0, 0] = ri_tensor[0, 0]
        empty_inst[0, 1] = mem[0, 0]

        for j in range(num_features):
            seed_idx = seed_label == (j + 1)
            pos_annot_c = torch.where(seed_idx, nuc, zero)
            neg_annot_c = torch.where(seed_idx, zero, nuc)

            if pos_annot_c.gt(0.5).float().sum() < 200: 
                continue

            empty_inst[0, 2] = pos_annot_c
            empty_inst[0, 3] = neg_annot_c

            out = cell_by_cell_net(empty_inst)
            out = F.interpolate(out, (args.preprocess['center_crop_size'], args.preprocess['zoomed_size'], args.preprocess['zoomed_size']), mode='nearest')
            out_inst[j] += out[0][0]

        try:
            max_val, max_idx = torch.max(out_inst, axis=0)
            max_idx += 1
            cell = max_idx.where(max_val >= 0.5, zero.long())
            cell = cell.cpu().numpy()
        except:
            cell = np.zeros((args.preprocess['center_crop_size'], args.preprocess['zoomed_size'], args.preprocess['zoomed_size']))

        resize_org = Resize((args.preprocess['center_crop_size'], ri_np.shape[1], ri_np.shape[2]), order=0)
        for key, value in img.items():
            if key in ['membrane', 'nucleus', 'nucleoli', 'lipid']:
                value /= 4
                mask = np.where(value > 0.5, 1, 0).astype(float)
                img[key] = resize_org(np.stack([mask]))[0]
                
                pad_data = np.zeros(ri_np.shape)
                z_offset = (ri_np.shape[0] - img[key].shape[0]) // 2
                pad_data[z_offset:z_offset + img[key].shape[0]] = img[key]
                pad_data = np.where(pad_data > 0.5, 1, 0).astype(float)
                img[key] = pad_data
    
        cell = resize_org(np.stack([cell]))[0]
        pad_data = np.zeros(ri_np.shape)
        z_offset = (ri_np.shape[0] - cell.shape[0]) // 2
        pad_data[z_offset:z_offset + cell.shape[0]] = cell
        cell = pad_data

        img['cell'] = cell
        img['ri'] = ri_np
        if not os.path.exists(args.save['path']):
            os.makedirs(args.save['path'])
        with h5py.File(os.path.join(args.save['path'], os.path.basename(file).replace(".TCF", "_") + time + '.hdf'), 'w') as hdf:
            for key, value in img.items():
                hdf.create_dataset(key, data=value, compression='gzip')