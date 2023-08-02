import platform
import platform
import sys
import torch
import random
from rknn.api import RKNN
from torch.nn.utils import prune
torch.manual_seed(1024)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Multi_Conv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 64, 3, 1, 1)
        self.conv2 = ConvBlock(64, 64, 1, 1, 1)
        self.conv3 = ConvBlock(64, 64, 2, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':

    # Generate pt model used to test
    model = Multi_Conv()
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=0.5, n=2, dim=0)
            prune.remove(m, 'weight')
    model.eval()
    fake_in = torch.randn(1, 3, 224, 224)
    jt_model = torch.jit.trace(model, fake_in)
    torch.jit.save(jt_model, 'Multi_Conv.pt')

    # Default target and device_id
    target = 'rv1126'
    device_id = None

    # Parameters check
    if len(sys.argv) == 1:
        print("Using default target rv1126")
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 3:
        target = sys.argv[1]
        device_id = sys.argv[2]
        print('Set target: {}, device_id: {}'.format(target, device_id))
    elif len(sys.argv) > 3:
        print('Too much arguments')
        print('Usage: python {} [target] [device_id]'.format(sys.argv[0]))
        print('Such as: python {} rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(-1)

    rknn = RKNN(verbose=True)

    rknn.config(model_pruning=True, target_platform=[target])

    print('--> Load pytorch model')
    ret = rknn.load_pytorch(model='Multi_Conv.pt', input_size_list=[[3,224,224]])
    if ret != 0:
        print("Load pytorch model failed.")
        rknn.release()
        exit(ret)
    print('done')
        
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='dataset.txt')
    if ret != 0:
        print("Build RKNN model failed.")
        rknn.release()
        exit(ret)

    print('--> Export RKNN model')
    ret = rknn.export_rknn('Multi_Conv_pruning.rknn')
    if ret != 0:
        print("Export RKNN model failed.")
        rknn.release()
        exit(ret)

    # Inference with PyTorch
    pt_output = model(fake_in)

    # Init RKNN runtime.
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime failed.')
        rknn.release()
        exit(ret)

    # Inference with RKNN
    rknn_input = fake_in.numpy().transpose(0,2,3,1)     # nchw -> nhwc
    rknn_output = rknn.inference([rknn_input])

    # Compare the inference results of PyTorch and RKNN.
    cos_sim = torch.cosine_similarity(pt_output.reshape(1, -1), torch.tensor(rknn_output[0]).reshape(1, -1))
    print("cos sim:", cos_sim[0].detach().numpy())

    rknn.release()

