import torch
from torch.autograd import Variable
import numpy as np

def demo(model, loader, viz=False, device=torch.device('cpu')):

    print('-' * 10 + 'evaluation' + '-' * 10)
    model.model.eval()
    model.inpainting.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if torch.cuda.is_available():
                img = Variable(data['img']).to(device)
            else:
                print('CUDA error')
                sys.exit(0)
            
            # forward
            m0, m1, m2, m3, mask, encoded = model.model(img)
            decoded = model.inpainting.head(encoded)

            # visualize
            if viz:
                model.viz_result(img, mask, decoded)

            # save results
            results = {}
            results.update(img=img.detach().cpu().numpy().astype(np.float32))
            results.update(mask=mask.detach().cpu().numpy().astype(np.float32))
            results.update(pred=decoded.detach().cpu().numpy().astype(np.float32))
            model.save_results(results, i)
            
            print('batch: %d/%d ' %(i, len(loader)))