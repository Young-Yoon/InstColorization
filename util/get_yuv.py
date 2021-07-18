import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

def get_pos_from_pidx(h, w, pi, bh, bw):
    b_row = w//bw
    b_fr = (h//bh)*b_row
    fi, bi = divmod(pi, b_fr)
    pos = fi * h * w * 3 // 2
    r, c = divmod(bi, b_row)
    r, c = r * bh, c * bw
    return pos, r, c

def get_pos_from_fridx(h, w, fi, bh, bw):
    b_row = w//bw
    b_fr = (h//bh)*b_row
    pi = fi * b_fr
    pos = fi * h * w * 3 // 2
    return pos, pi

def yuv2np(fn, img, patch, ref=0, bit_out=8):
    """
    fn: yuv file path in full
    img={'w','h','bit'}
    patch={'idx':index in video,'blk':patch size}
    Y,U,V: 2d numpy array
    """
    h, w, byte_size = img['h'], img['w'], 2 if img['bit']>8 else 1
    srctype, dsttype = list(map(lambda x: np.int16 if x>8 else np.uint8, [img['bit'], bit_out]))
    # print(srctype, dsttype)

    # print(fn, "yuv h w byte", h, w, byte_size)
    pi, bh, bw = patch['idx'], patch['blk'], patch['blk']
    pos_fr, r, c = get_pos_from_pidx(h, w, pi, bh, bw)
    
    Y, U, V = [], [], []
    if ref > 0:
        Utop, Uleft, Vtop, Vleft = [], [], [], []
    pos_y, pos_u = pos_fr + w * r + c, pos_fr + w * h + (w // 2) * (r // 2) + c // 2 
    pos_v = pos_u + w * h // 4
    
    with open(fn, 'rb') as stream:
        # print("np type bit byte", dtype, img['bit'], byte_size)
        def readblock(pos, stride, blkh, blkw):
            stream.seek(pos * byte_size, 0)
            out = np.fromfile(stream, dtype=srctype, count=blkw).reshape(1, blkw)
            for i in range(1, blkh):
                stream.seek((stride - blkw) * byte_size, 1)
                out = np.append(out, np.fromfile(stream, dtype=srctype, count=blkw).reshape(1, blkw), axis=0)
            if img['bit'] > bit_out: 
                out = (out//(2**(img['bit']-bit_out))).astype(dsttype)
            elif img['bit'] < bit_out:
                out = out.astype(dsttype)*(2**(bit_out-img['bit']))
            # print(out.dtype, out[:3,:3])
            return out

        Y = readblock(pos_y, w, bh, bw)
        U = readblock(pos_u, w//2, bh//2, bw//2)
        V = readblock(pos_v, w//2, bh//2, bw//2)
        if ref > 0:
            if r == 0:
                Utop = np.multiply(np.ones((1, bw//2), dtype=np.int16), 512)
                Vtop = np.multiply(np.ones((1, bw//2), dtype=np.int16), 512)
            else:
                Utop = readblock(pos_u - w//2, w//2, 1, bw//2)
                Vtop = readblock(pos_v - w//2, w//2, 1, bw//2)
            if c == 0: # Yleft=np.multiply(np.ones((bh, 2),dtype=np.int16),512)
                Uleft = np.multiply(np.ones((bh//2, 1), dtype=np.int16), 512)
                Vleft = np.multiply(np.ones((bh//2, 1), dtype=np.int16), 512)                
            else:      # Yleft= readblock(pos_y, w, bh, 2)
                Uleft = readblock(pos_u - 1, w // 2, bh // 2, 1)
                Vleft = readblock(pos_v - 1, w // 2, bh // 2, 1)
            Uref = np.vstack((Utop, np.transpose(Uleft)))  # np(2,W)
            Vref = np.vstack((Vtop, np.transpose(Vleft)))
            hasRef = np.array([1.0 if r > 0 else 0.0, 1.0 if c > 0 else 0.0])
    '''
    print(fn, "idx(img,patch,blk)", img_idx, patch_idx, blk_idx, "WHFRC", w, h, fr, r, c)
    #plt_yuvC(Y,U,V,chfmt='RGB') 
    plt_yuvC(np.hstack((Yleft,Y)),np.hstack((Uleft,U)),np.hstack((Vleft,V)),chfmt='RGB') #'''
    if ref > 0:
        return Y, U, V, Uref, Vref, hasRef
    else: 
        return Y, U, V
    

PSNR8 = lambda x: 10 * np.log10(255 ** 2 / x)
PSNR10 = lambda x: 10 * np.log10(1023 ** 2 / x)
    
def yuv2bgr(yuv, chfmt='BGR'):
    YUV2BGR = np.array([[1.164, 2.017, 0.000],  # B
                        [1.164, -0.392, -0.813],  # G
                        [1.164, 0.000, 1.596]])  # R
    yuv = yuv.astype(np.float)
    yuv[:, :, 0] = yuv[:, :, 0] - 16.0  # 0.0625   # Offset Y by 16
    yuv[:, :, 1:] = yuv[:, :, 1:] - 128.0  # 0.5  # Offset UV by 128

    transform = YUV2BGR.T if chfmt == 'BGR' else YUV2BGR[::-1].T
    
    # print(transform.dot(transform.T))
    return yuv.dot(transform).clip(0, 255).astype(np.uint8)


# Helper Functions
def yuv2uint8(Y, U, V, in_bit):
    if isinstance(Y.flat[0], np.int16):
        scale = 2**(in_bit-8)
        Y, U, V = list(map(lambda x: ((x.astype(np.float)+scale/2)/scale).astype(np.uint8), [Y, U, V]))
    return Y, U, V

def yuv2bgr_cv(Y, U, V, chfmt='BGR'):
    cv2color = cv2.COLOR_YUV2BGR_I420 if chfmt=='BGR' else cv2.COLOR_YUV2RGB_I420
    return cv2.cvtColor(np.hstack((Y.ravel(), U.ravel(), V.ravel()))
                        .reshape((Y.shape[0]*3//2, Y.shape[1])), cv2color)
                     

def plt_yuvC(Y, U, V, in_bit=10, chfmt='BGR', orig=None, 
             savefig=None, info=('', '', ''), onlyC=False, figsize=(16,8)):
    # print("Yin: ", Y.flat[:4])
    # Display format: uint8
    Y, U, V = yuv2uint8(Y, U, V, in_bit)
    # print("Yuint8: ", Y.flat[:4], "U", U[:3, :4], "V", V[:3, :4])

    if orig:  # orig={'Y','U','V','in_bit','chfmt'}
        orig['Y'], orig['U'], orig['V'] = yuv2uint8(orig['Y'], orig['U'], orig['V'], orig['in_bit'])
        
    # Convert: YCrCb2BGR(Dull) <-> YUV2BGR_I420=yuv2bgr(Colorful)
    # yuv: 444 in uint8
    UVhalf = np.append(np.expand_dims(U, 2), np.expand_dims(V, 2), axis=2)
    UV = UVhalf.repeat(2, axis=0).repeat(2, axis=1)    
    yuv = np.append(np.expand_dims(Y, 2), UV, axis=2)
    clr = yuv2bgr(yuv, chfmt=chfmt)    
    
    ydim, yuvdim = (Y.shape[1], Y.shape[0]), (Y.shape[0]*3//2, Y.shape[1])
    yvu = cv2.merge((Y, cv2.resize(V, ydim), cv2.resize(U, ydim))) 
    bgr1 = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR) # dull color
    
    # yuvdata = np.hstack((Y.ravel(), U.ravel(), V.ravel())).reshape(yuvdim)
    # bgr2 = cv2.cvtColor(yuvdata, cv2.COLOR_YUV2BGR_I420)          
    bgr2 = yuv2bgr_cv(Y, U, V, chfmt=chfmt)
    # print("bgr1: ", bgr1[:2,:4,:], "\nbgr2: ", bgr2[:2,:4,:], "\nclr :", clr[:2,:4,::-1]) 
    # cv2.imshow("bgr", cv2.hconcat([bgr1, bgr2, clr[:,:,::-1]]))
    
    # Convert: BGR2YUV_I420
    yuv_rec = cv2.cvtColor(bgr2, cv2.COLOR_BGR2YUV_I420)
    uv_rec = np.transpose(yuv_rec[Y.shape[0]:, :].reshape((2, Y.shape[0]//2, Y.shape[1]//2)), (1,2,0))
    
    # Color Conversion Errors 
    mse_uv = calcMSE(uv_rec, UVhalf)
    psnr_uv = list(map(PSNR8, mse_uv))
    # print(mse_uv, psnr_uv)
    
    if orig:
        clr_org = yuv2bgr_cv(orig['Y'], orig['U'], orig['V'], chfmt=chfmt)
        uv_org = np.append(np.expand_dims(orig['U'], 2), np.expand_dims(orig['V'], 2), axis=2)
        # Caculate Errors
        mse_uv = calcMSE(UVhalf, uv_org)
        psnr_uv = list(map(PSNR8, mse_uv))
        info = (info[0], info[1]+"U MSE {:.1f} -> {:.2f} dB".format(mse_uv[0], psnr_uv[0]), info[2]+"V MSE {:.1f} -> {:.2f} dB".format(mse_uv[1], psnr_uv[1]))
    

    # print(list(map(lambda img: (np.min(img), np.max(img)), [Y,U,V])))
    fig = plt.figure(num=None, figsize=figsize)  # plt.rcParams["figure.figsize"] = figsize
    if onlyC == False:
        # fig, axs = plt.subplots(1,4)
        if orig:
            im_list = [(np.append(orig['Y'], Y, axis=0), info[0]), 
                       (np.append(orig['U'], U, axis=0), info[1]),
                       (np.append(orig['U'], V, axis=0), info[2]) ]
        else:
            im_list = [(Y, info[0]), (U, info[1]), (V, info[2])]
            
        for i, s in enumerate(im_list):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(s[0], cmap='gray', vmin=0, vmax=255)
            ax.title.set_text("{}".format(s[1]))
        ax = fig.add_subplot(144)
    else:
        ax = fig.add_subplot(111)
        ax.title.set_text(info[0])
    if orig:
        ax.imshow(np.append(clr_org, clr, axis=0), vmin=0, vmax=255)
    else:
        ax.imshow(clr, vmin=0, vmax=255)
    '''
    axs[0].imshow((Y//4).clip(0,255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    axs[1].imshow((U//4).clip(0,255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    axs[2].imshow((V//4).clip(0,255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)       
    axs[3].imshow(CLR.clip(0,255).astype(np.uint8), vmin=0, vmax=255)
    '''
    if savefig: plt.savefig(savefig)
    plt.show()

    return 0 if not orig else mse_uv


def calcMSE(outputs, targets):
    ''' in: numpy(WHC)
    outputs, targets: numpy(32,32,2):WHC
    MSE(mean square error): U, V
    '''
    outputs, targets = outputs.astype(np.float), targets.astype(np.float)
    flattenUV = lambda x: (x[:, :, 0].flatten(), x[:, :, 1].flatten())
    difU, difV = flattenUV(np.subtract(outputs, targets))
    dim = difU.size
    MSE = lambda x: np.dot(x, x) / dim

    # get_range = lambda x: (np.min(x), np.max(x))
    # print(dim, get_range(difU), get_range(difV))
    return MSE(difU), MSE(difV)


# Dataset statistics
meanBVI, stdBVI = [384.2, 491.0, 526.5], [225.1, 498.1, 531.4]  # [511.5]*3, [511.5]*3
meanBVIinv, stdBVIinv = np.multiply(np.true_divide(meanBVI, stdBVI), -1.0), np.true_divide(1, stdBVI)
# numpy <-> Tensor
np2tensor = transforms.Lambda(lambda img: torch.from_numpy(img.astype(np.float32)).unsqueeze(0))
uv2tensor = transforms.Lambda(lambda img: torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1))
tensor2np = transforms.Lambda(lambda x: np.clip(torch.squeeze(x).numpy(), 0.0, 1023.0).astype(np.int16))
tensor2uv = transforms.Lambda(lambda x: np.clip(torch.squeeze(x).permute(1, 2, 0).numpy(), 0.0, 1023.0).
                              astype(np.int16))
# transforms
Y_transform = transforms.Compose([np2tensor, transforms.Normalize(meanBVI[0], stdBVI[0]), ])
UV_transform = transforms.Compose([uv2tensor, transforms.Normalize(meanBVI[1:], stdBVI[1:]), ])
Y_transformInv = transforms.Compose([transforms.Normalize(meanBVIinv[0], stdBVIinv[0]), tensor2np, ])
UV_transformInv = transforms.Compose([transforms.Normalize(meanBVIinv[1:], stdBVIinv[1:]), tensor2uv, ])

class VVC_SDR(Dataset):
    def __init__(self, img_dir, patch=64, test_set=True):
        # super(VVC_SDR, self).__init__()
        self.img_dir = img_dir  # to_path/Videos
        self.img_names = sorted(filter(lambda x: x.endswith('.yuv'), os.listdir(img_dir)))        
        vvc_10bits = {'Campf', 'CatRo', 'Dayli', 'FoodM', 'Marke', 'ParkR', 'Ritua', 'Tango'}
        self.bits = [10 if fn[:5] in vvc_10bits else 8 for fn in self.img_names]
        self.patch, self.I = patch, 1
             
        num_patch, width, height = [0], [], []
        for i, fn in enumerate(self.img_names):
            wxh = fn.split('_')[1].split('x')
            w, h = int(wxh[0]), int(wxh[1])
            width.append(w), height.append(h)
            filesize = os.path.getsize(img_dir+'/'+fn)
            bytesize = 2 if self.bits[i]>8 else 1
            num_fr = filesize//(w*h*bytesize*3//2) if not test_set else 1
            num_fr = 1 + (num_fr-1)//self.I
            patch_fr = (w // patch) * (h // patch)
            num_patch.append(num_patch[-1] + patch_fr * num_fr)
            # print(f"{fn:55} {w:4d}x{h:4d}x{num_fr:3d} {filesize:11d} {patch_fr*num_fr:7d} {num_patch[-1]:7d}")

        self.num_patch = num_patch
        self.info = {'width': width, 'height': height}

        self.transform = Y_transform
        self.target_transform = UV_transform
        self.transformInv = Y_transformInv
        self.target_transformInv = UV_transformInv
        # self.calcStat()
        # print(self.img_names, self.info, self.num_patch)
    

    def __len__(self):
        return self.num_patch[-1]
    
    def __getitem__(self, idx, bit_out=10, out_path=None, debug=False):
        img_idx = np.searchsorted(self.num_patch[1:], idx, side='right')
        img_path = os.path.join(self.img_dir, self.img_names[img_idx])
        img_info = {'h': self.info['height'][img_idx], 'w':self.info['width'][img_idx], 'bit':self.bits[img_idx]}
        patch_idx = idx - self.num_patch[img_idx]
        # precision = self.bits if isinstance(self.bits,int)==1 else self.bits[img_idx]
        patch_info = {'idx':patch_idx, 'blk':self.patch}

        Y, U, V, Uref, Vref, hasRef = yuv2np(img_path, img_info, patch_info, ref=1, bit_out=bit_out)
        image = Y
        label = np.dstack((U, V))  # np.append(np.expand_dims(U,axis=2), np.expand_dims(V,axis=2), axis=2)
        refchroma = np.dstack((Uref, Vref))
        hasRef = torch.from_numpy(hasRef)        
        if debug:
            print(image.size, label.size)

        # print('Before Transform Y,U,V:', list(map(lambda x:(np.min(x),np.max(x)),[image, U, V])))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            refchroma = self.target_transform(refchroma)

        data = {"image": image, "label": label, "refC": refchroma, "refExist": hasRef}
        # ''' image: Tensor(1,1,H,W)->np(H,W), label: Tensor(1,2,H//2,W//2)->np(H//2,W//2,2)
        if debug:
            # print('Range After:', list(map(lambda x: (np.min(x.numpy()), np.max(x.numpy())), [image, label])))
            image, label = self.transformInv(image), self.target_transformInv(label)
            # print(image.size, label.size)
            # print('Range Restore:', list(map(lambda x: (np.min(x), np.max(x)), [image, label])))
            plt_yuvC(image, np.squeeze(label[:, :, 0]), np.squeeze(label[:, :, 1]), 
                     in_bit=bit_out, chfmt='RGB',
                     info=(str(img_idx) + ':' + self.img_names[img_idx], str(patch_idx), ''))  # '''
        # self.calcImageStat(img_path, img_idx) #'''
        
        if out_path:
            img_file = self.patch_filename(img_idx, patch_idx)
            img_path = os.path.join(out_path, img_file)
            gray_img = cv2.cvtColor(Y, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(img_path, gray_img)
            
        return data    
    
    def patch_filename(self, img_idx, patch_idx):
        return self.img_names[img_idx][:-4]+'_'+str(patch_idx)+'.png'
    
    def generate_frame(self, res_dir, img_idx, fr=0):
        h, w = self.info['height'][img_idx], self.info['width'][img_idx]
        bh = bw = self.patch        
        b_row = w//bw
        b_fr = (h//bh)*b_row
        
        mseU, mseV = 0, 0
        im_tile = []
        for j in range(h//bh):
            im_list = []
            for i in range(w//bw):
                patch_idx = fr*b_fr + j*b_row + i
                img_path = os.path.join(res_dir, self.patch_filename(img_idx, patch_idx))
                cv_mat = cv2.imread(img_path)
                cv_mat = cv2.resize(cv_mat, (bw, bh), interpolation=cv2.INTER_CUBIC)
                im_list.append(cv_mat)
                
                pred = np.asarray(cv_mat)
                pred_yuv = cv2.cvtColor(pred, cv2.COLOR_BGR2YUV_I420)
                pred_uv = np.transpose(pred_yuv[bh:, :].reshape((2, bh//2, bw//2)), (1,2,0))
                
                orig = self.__getitem__(self.num_patch[img_idx]+patch_idx)
                Y = self.transformInv(orig['image'])
                UV = self.target_transformInv(orig['label'])
                # print("Pred:",pred.shape, "Orig:",UV.shape, UV[:2,:3,0])
                
                org_yuv = {'Y':Y, 'U':np.squeeze(UV[:,:,0]), 'V':np.squeeze(UV[:,:,1]), 'in_bit':10}
                mse_uv = plt_yuvC(pred_yuv[:bh,:], np.squeeze(pred_uv[:,:,0]), np.squeeze(pred_uv[:,:,1]), 
                         chfmt='RGB', in_bit=8, orig=org_yuv,
                         info=(str(img_idx) + ':' + self.img_names[img_idx], str(patch_idx), ''))
                
                mseU += mse_uv[0]
                mseV += mse_uv[1]
                psnr_uv = list(map(PSNR8, mse_uv))
                print("Patch {:2d} U MSE {:.1f} -> {:.2f} dB  V MSE {:.1f} -> {:.2f} dB".format(patch_idx, mse_uv[0], psnr_uv[0], mse_uv[1], psnr_uv[1]))
                                
            im_tile.append(cv2.hconcat(im_list))
        im_frame = cv2.vconcat(im_tile)
        
        mseU /= b_fr
        mseV /= b_fr
        psnr_uv = list(map(PSNR8, [mseU, mseV]))
        print("Frame {:2d} U MSE {:.1f} -> {:.2f} dB  V MSE {:.1f} -> {:.2f} dB".format(fr, mseU, psnr_uv[0], mseV, psnr_uv[1]))
        

        merged_filename = os.path.join(res_dir, self.patch_filename(img_idx, ""))
        cv2.imwrite(merged_filename, im_frame)    
            
        
def get_pos_from_fridx(h, w, fi, bh, bw):
    b_row = w//bw
    b_fr = (h//bh)*b_row
    pi = fi * b_fr
    pos = fi * h * w * 3 // 2
    return pos, pi
        
