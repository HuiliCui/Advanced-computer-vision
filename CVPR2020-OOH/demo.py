import torch
import os
from cmd_parser import parse_config
from modules import init, ImageLoader, ModelLoader
from process import demo

def main(**args):

    # global settings
    dtype = torch.float32
    batchsize = args.pop('batchsize')
    workers = args.pop('worker')
    device = torch.device(index=args.pop('gpu_index'),type='cuda')
    viz = args.pop('viz')

    # init project settings
    out_dir, logger, smpl, generator, occlusions = init(dtype=dtype, **args)

    # create data loader
    dataset = ImageLoader(**args)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    # load model
    model = ModelLoader(device=device, output=out_dir, smpl=smpl, generator=generator, **args)

    # testing mode
    demo(model, test_loader, viz=viz, device=device)
    
    logger.close()

if __name__ == "__main__":
    args = parse_config()
    main(**args)
