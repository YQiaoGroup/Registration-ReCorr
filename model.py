import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import UpdateBlock
from extractor import Encoder
from local_search import search, warp
from transform import SpatialTransformer


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class ReCorr(nn.Module):
    def __init__(self, iters={'16x': 3, '8x': 3, '4x': 2, '2x': 2}, mixed_precision=False, test_mode=False, diff=False):
        super(ReCorr, self).__init__()

        self.iters = iters
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode
        self.diff = diff

        self.hidden_dim = 16
        self.context_dim = 16

        self.dropout = 0

        self.fnet = Encoder(output_dim=32, bn=True)

        self.update_block_16 = UpdateBlock(hidden_dim=self.hidden_dim, input_dim=self.context_dim)
        self.update_block_8 = UpdateBlock(hidden_dim=self.hidden_dim, input_dim=self.context_dim)
        self.update_block_4 = UpdateBlock(hidden_dim=self.hidden_dim, input_dim=self.context_dim)
        self.update_block_2 = UpdateBlock(hidden_dim=self.hidden_dim, input_dim=self.context_dim)
        self.flow_head = nn.Conv3d(16+16+3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.stn = SpatialTransformer(mode="bilinear")


    def forward(self, image1, image2, test_mode=False):
        test_mode = self.test_mode or test_mode
        if not test_mode:
            flow_list = self.compute_flow(image1, image2)
            moved_list = self.warp_list(flow_list, image2)
            return flow_list, moved_list
        else:
            flow = self.inference_flow(image1, image2)
            moved = self.warp_list([flow], image2)

            return flow, moved[0]
    
    def _compute_stage_flow(self, flow, scale, fmap1, fmap2, iters, update_block):

        with autocast(enabled=self.mixed_precision):
            hidden, context = torch.split(fmap1, [self.hidden_dim, self.context_dim], dim=1)
            hidden = torch.tanh(hidden)
            context = torch.relu(context)

        flow_predictions = []
        for itr in range(iters):
            flow = flow.detach()
            out_corrs = search(fmap1, fmap2, flow)

            with autocast(enabled=self.mixed_precision):
                hidden, delta_flow = update_block(hidden, context, out_corrs, flow)

            flow = flow + delta_flow
            if self.diff:
                flow = self.svf(flow)
            flow_up = self.upflow(flow, scale=scale)
            flow_predictions.append(flow_up)

        return flow, flow_predictions


    def compute_flow(self, image1, image2):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.mixed_precision):
            fmap1_pyramid, fmap2_pyramid = self.fnet([image1, image2])

        N, _, D, H, W = fmap1_pyramid['16x'].shape
        flow = torch.zeros(N, 3, D, H, W).to(image1.device)
        flow_predictions = []

        # 1/16
        flow, stage_preds = self._compute_stage_flow(flow=flow, scale=16,
                                                    fmap1=fmap1_pyramid['16x'], 
                                                    fmap2=fmap2_pyramid['16x'],
                                                    iters=self.iters['16x'], 
                                                    update_block=self.update_block_16)
        flow_predictions = flow_predictions + stage_preds
        # 1/8
        flow = self.upflow(flow, scale=2)
        flow, stage_preds = self._compute_stage_flow(flow=flow, scale=8,
                                                    fmap1=fmap1_pyramid['8x'], 
                                                    fmap2=fmap2_pyramid['8x'],
                                                    iters=self.iters['8x'], 
                                                    update_block=self.update_block_8)
        flow_predictions = flow_predictions + stage_preds

        # 1/4
        flow = self.upflow(flow, scale=2)
        flow, stage_preds = self._compute_stage_flow(flow=flow, scale=4,
                                                    fmap1=fmap1_pyramid['4x'], 
                                                    fmap2=fmap2_pyramid['4x'],
                                                    iters=self.iters['4x'], 
                                                    update_block=self.update_block_4)
        flow_predictions = flow_predictions + stage_preds

        # 1/2
        flow = self.upflow(flow, scale=2)
        flow, stage_preds = self._compute_stage_flow(flow=flow, scale=2,
                                                    fmap1=fmap1_pyramid['2x'], 
                                                    fmap2=fmap2_pyramid['2x'],
                                                    iters=self.iters['2x'], 
                                                    update_block=self.update_block_2)
        flow_predictions = flow_predictions + stage_preds

        # 1/1 - final stage
        flow = self.upflow(flow, scale=2)
        fmap1 = fmap1_pyramid['1x']
        fmap2 = fmap2_pyramid['1x']
        disp = self.flow_head(torch.cat([fmap1, fmap2, flow], dim=1))
        flow_predictions.append(disp)

        return flow_predictions
    
    def _inference_stage_flow(self, flow, fmap1, fmap2, iters, update_block):

        with autocast(enabled=self.mixed_precision):
            hidden, context = torch.split(fmap1, [self.hidden_dim, self.context_dim], dim=1)
            hidden = torch.tanh(hidden)
            context = torch.relu(context)

        for itr in range(iters):
            flow = flow.detach()
            out_corrs = search(fmap1, fmap2, flow)

            with autocast(enabled=self.mixed_precision):
                hidden, delta_flow = update_block(hidden, context, out_corrs, flow)

            flow = flow + delta_flow
            if self.diff:
                flow = self.svf(flow)

        return flow
    
    def inference_flow(self, image1, image2):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.mixed_precision):
            fmap1_pyramid, fmap2_pyramid = self.fnet([image1, image2])

        N, _, D, H, W = fmap1_pyramid['16x'].shape
        flow = torch.zeros(N, 3, D, H, W).to(image1.device)

        # 1/16
        flow = self._inference_stage_flow(flow=flow, 
                                        fmap1=fmap1_pyramid['16x'], 
                                        fmap2=fmap2_pyramid['16x'],
                                        iters=self.iters['16x'], 
                                        update_block=self.update_block_16)

        # 1/8
        flow = self.upflow(flow, scale=2)
        flow = self._inference_stage_flow(flow=flow, 
                                        fmap1=fmap1_pyramid['8x'], 
                                        fmap2=fmap2_pyramid['8x'],
                                        iters=self.iters['8x'], 
                                        update_block=self.update_block_8)

        # 1/4
        flow = self.upflow(flow, scale=2)
        flow = self._inference_stage_flow(flow=flow, 
                                        fmap1=fmap1_pyramid['4x'], 
                                        fmap2=fmap2_pyramid['4x'],
                                        iters=self.iters['4x'], 
                                        update_block=self.update_block_4)

        # 1/2
        flow = self.upflow(flow, scale=2)
        flow = self._inference_stage_flow(flow=flow, 
                                        fmap1=fmap1_pyramid['2x'], 
                                        fmap2=fmap2_pyramid['2x'],
                                        iters=self.iters['2x'], 
                                        update_block=self.update_block_2)

        # 1/1 - final stage
        flow = self.upflow(flow, scale=2)
        fmap1 = fmap1_pyramid['1x']
        fmap2 = fmap2_pyramid['1x']
        disp = self.flow_head(torch.cat([fmap1, fmap2, flow], dim=1))

        return disp


    def warp_list(self, flow_list, moving):
        moved_list = []
        for flow in flow_list:
            moved_list.append(self.stn(moving, flow))
        return moved_list
    

    def upflow(self, flow, mode='trilinear', scale=8):
        new_size = (scale * flow.shape[2], scale * flow.shape[3], scale * flow.shape[4])
        if mode == 'nearest':
            return scale * F.interpolate(flow, size=new_size, mode=mode)
        else:
            return scale * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
        
        
    def svf(self, flow, scale=1, steps=7):
        disp = flow * (scale / (2 ** steps))
        for _ in range(steps):
            disp = disp + warp(disp, disp)
        return disp
    
