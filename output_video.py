import utils
import torch
import numpy as np
from net_stdf import MFVQE #导入你使用的网络结构
import re

#raw_yuv_path = '/home/rhh/STDF-PyTorch/MFQEv2_dataset/test_18/raw/yoyo_1920x1080_232_raw.yuv' #example
raw_yuv_path = '/path/to/your/raw_video.yuv'
#lq_yuv_path = '/home/rhh/STDF-PyTorch/MFQEv2_dataset/test_18/HM16.5_LDPcap/QP37/yoyo_1920x1080_232_QP37.yuv' #example
lq_yuv_path = '/path/to/your/lq_video.yuv'
matchinfo = re.search(r"_(\d+)x(\d+)_(\d+)\.yuv$", raw_yuv_path)
w, h, nfs = int(matchinfo.group(1)),int(matchinfo.group(2)),int(matchinfo.group(3))
matchvideoname = re.search(r"([^/]+)(?=\.[^.]+$)",lq_yuv_path)
output_file_name = f"{matchvideoname}_enhanced.yuv" # 输出文件名"xxx_enhanced.yuv"
ckp_path = 'path/to/your/pth' #存放的模型文件
def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {
        'radius': 3,
        'stdf': {
            'in_nc': 1,
            'out_nc': 64,
            'nf': 32,
            'nb': 3,
            'base_ks': 3,
            'deform_ks': 3,
            },
        'qenet': {
            'in_nc': 64,
            'out_nc': 1,
            'nf': 48,
            'nb': 8,
            'base_ks': 3,
            },
        } #这里根据自己的设置来 我是可视化的stdf的增强后效果
    model = MFVQE(opts_dict=opts_dict)
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    lq_y ,lq_u, lq_v = utils.import_yuv(
            seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)
    
    # 会追加，我懒得写assert了，如果文件已经存在每次使用前要删除，不然会接着写在后面
    fp = open(output_file_name, 'ab')

    for idx in range(nfs):
        with torch.no_grad():
            # load lq
            idx_list = list(range(idx-3,idx+4))
            idx_list = np.clip(idx_list, 0, nfs-1)
            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()

            # write enhanced frame
            enhanced_frm = model(input_data) # 得到增强后y帧
            enhanced_y = torch.clamp(enhanced_frm, 0, 1) # 避免出现(0,1)之外的值
            enhanced_frm_255 = torch.round(torch.squeeze(enhanced_y*255)).to(dtype =torch.uint8).cpu().numpy()
            enhanced_frm_255 = enhanced_frm_255.ravel()
            tmp_lq_u = lq_u[idx].ravel() # 使用压缩视频的u和v
            tmp_lq_v = lq_v[idx].ravel()
            first_frm = np.concatenate((enhanced_frm_255, tmp_lq_u, tmp_lq_v))
            # first_frm = first_frm.astype(np.uint8)
            fp.write(first_frm.tobytes())
            print(f"write {idx} frame")

    fp.close()
    print(f"output video name:{output_file_name}")
    print('> done.')


if __name__ == '__main__':
    main()

