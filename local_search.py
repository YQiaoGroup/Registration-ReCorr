import torch
import torch.nn.functional as F


def coords_grid(batch, d, h, w, device):
    coords = torch.meshgrid([torch.arange(d, device=device),
                             torch.arange(h, device=device),
                             torch.arange(w, device=device)])
    coords = torch.stack(coords, dim=0).float() # [z, x, y] → [y, x, z], (3, d, h, w)
    return coords[None].repeat(batch, 1, 1, 1, 1)


def warp(disp, image):
    N, _, D, H, W = image.shape
    temp_coord = coords_grid(N, D, H, W, device=image.device)
    coords = temp_coord + disp # (batch, 3, D, H, W)

    coords[:, 0, :, :, :] = 2.0 *coords[:, 0, :, :, :].clone() / max(D - 1, 1) - 1.0
    coords[:, 1, :, :, :] = 2.0 *coords[:, 1, :, :, :].clone() / max(H - 1, 1) - 1.0
    coords[:, 2, :, :, :] = 2.0 *coords[:, 2, :, :, :].clone() / max(W - 1, 1) - 1.0
    
    coords = coords.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]] # (z,y,x) → (x,y,z)
    output = F.grid_sample(image, coords, align_corners=True, padding_mode="border")
    return output


def search(fmap1, fmap2, flow, radius = 3, scale = 1):
    ''' init '''
    # (b, C, D, H, W) → (b, 1, D, H, W) 
    N, C, D, H, W = fmap1.shape
    ''' warp '''
    fmap2_warp = warp(flow, fmap2)
    cost_volume = []

    ''' search '''
    padd_map2 = F.pad(fmap2_warp, [radius//2, ]*6, mode = 'replicate')
    # random search
    for i in range(radius):
        for j in range(radius):
            for k in range(radius):
                map2 = padd_map2[:, :, k:k+D, j:j+H, i:i+W] # shift left, up, back
                cost = torch.mean(fmap1 * map2, dim=1, keepdim=True) # (b, C, D, H, W) → (b, 1, D, H, W)
                cost_volume.append(cost)

    out_corrs = torch.cat(cost_volume, dim=1) # (b, r^3, D, H, W)
    return out_corrs
