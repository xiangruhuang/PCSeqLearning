




class PointNet2RepSurf(nn.Module):
    def __init__(self, runtime_cfg, model_cfg, **kwargs):
        super(PointNet2RepSurf, self).__init__()
        input_channels = runtime_cfg.get("num_point_features", None)
        self.input_key = runtime_cfg.get("input_key", 'point')
        
        return_polar = model_cfg.get("RETURN_POLAR", False)
        T = model_cfg.get("SCALE", 1)
        sa_channels = model_cfg["SA_CHANNELS"]
        fp_channels = model_cfg["FP_CHANNELS"]
        num_sectors = model_cfg["NUM_SECTORS"]
        num_neighbors = model_cfg.get("NUM_NEIGHBORS", 32)
        self.output_key = model_cfg.get("OUTPUT_KEY", None)
        self.sa_modules = nn.ModuleList()

        cur_channel = input_channels - 3
        channel_stack = []
        for i, sa_channel in enumerate(sa_channels):
            sa_channel = [c*T for c in sa_channel]
            sa_module = PointNetSetAbstractionCN2Nor(4, num_neighbors, cur_channel, sa_channel, return_polar, num_sectors=num_sectors[i])
            self.sa_modules.append(sa_module)
            channel_stack.append(cur_channel)
            cur_channel = sa_channel[-1]

        self.fp_modules = nn.ModuleList()
        for i, fp_channel in enumerate(fp_channels):
            fp_channel = [c*T for c in fp_channel]
            fp_module = PointNetFeaturePropagationCN2(cur_channel, channel_stack.pop(), fp_channel)
            self.fp_modules.append(fp_module)
            cur_channel = fp_channel[-1]

        self.num_point_features = cur_channel

    def convert_to_bxyz(self, pos_feat_off):
        xyz = pos_feat_off[0]
        offset = pos_feat_off[2]
        batch_idx = []
        for i, offset in enumerate(offset):
            batch_idx.append(torch.full(xyz.shape[:1], i).long().to(xyz.device))
        batch_idx = torch.stack(batch_idx, -1)
        bxyz = torch.cat([batch_idx, xyz], dim=-1)

        return bxyz

    def forward(self, batch_dict):
        pos = batch_dict[f'{self.input_key}_bxyz'][:, 1:4].contiguous()
        feat = batch_dict[f'{self.input_key}_feat']
        batch_index = batch_dict[f'{self.input_key}_bxyz'][:, 0].round().long()
        num_points = []
        for i in range(batch_dict['batch_size']):
            num_points.append((batch_index == i).sum().int())
        num_points = torch.tensor(num_points).int().cuda()
        offset = num_points.cumsum(dim=0).int()

        data_stack = []
        pos_feat_off = [pos, feat, offset]
        data_stack.append(pos_feat_off)
        for i, sa_module in enumerate(self.sa_modules):
            pos_feat_off = sa_module(pos_feat_off)
            data_stack.append(pos_feat_off)
            key = f'pointnet2_sa${len(self.sa_modules)-i}_out'
            batch_dict[f'{key}_bxyz'] = self.convert_to_bxyz(pos_feat_off)
            batch_dict[f'{key}_feat'] = pos_feat_off[1]

        pos_feat_off = data_stack.pop()
        for i, fp_module in enumerate(self.fp_modules):
            pos_feat_off_cur = data_stack.pop()
            pos_feat_off_cur[1] = fp_module(pos_feat_off_cur, pos_feat_off)
            pos_feat_off = pos_feat_off_cur
            key = f'pointnet2_fp${i}_out'
            batch_dict[f'{key}_bxyz'] = self.convert_to_bxyz(pos_feat_off)
            batch_dict[f'{key}_feat'] = pos_feat_off[1]

        if self.output_key is not None:
            batch_dict[f'{self.output_key}_bxyz'] = self.convert_to_bxyz(pos_feat_off)
            batch_dict[f'{self.output_key}_feat'] = pos_feat_off_cur[1]

        return batch_dict
