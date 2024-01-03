import torch
import numpy as np
import abc, tqdm

mae_crit = torch.nn.L1Loss()

class BaseTrainer:
    def train(self, dataloader):
        train_loss = 0
        self.model.train()
        for batch in dataloader:
#        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='train'):
            self.opt.zero_grad()
            out = self._iter_step(batch)

            loss = mae_crit(out[0], out[1])
            loss.backward()
            self.opt.step()
            
            train_loss += loss.item()
        return train_loss / len(dataloader)
    
    def test(self, dataloader):
        test_loss = 0
        outputs   = {}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
#            for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='test'):
                out = self._iter_step(batch)    
                test_loss += mae_crit(out[0], out[1]).item()
                outputs = self._get_outputs(batch, out, outputs)
        outputs = self._parse_outputs(outputs)
        return test_loss / len(dataloader), outputs
    
    def pred(self, dataloader):
        outputs = {}
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
            #for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc='pred'):
                out = self._iter_step(batch)
                outputs = self._get_outputs(batch, out, outputs)
        outputs = self._parse_outputs(outputs)
        return outputs
    
    @abc.abstractmethod
    def _iter_step(self, batch):
        pass

    @abc.abstractmethod
    def _get_outputs(self, batch, output, outputs):
        pass

    @abc.abstractmethod
    def _parse_outputs(self, outputs):
        pass

class DBPNTrainer(BaseTrainer):
    def __init__(self, model, opt, residual=True, device=torch.device('cuda')):
        self.model = model
        self.opt   = opt
        self.residual = residual
        self.device = device

    def _iter_step(self, batch):
        inp, tgt, bic, _ = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        bic = bic.to(self.device)
        
        pred = self.model(inp)
        if self.residual:
            pred += bic
        return pred, tgt

    def _get_outputs(self, batch, output, outputs):
        _, tgt, bic, info = batch
        pred_img = output[0].cpu().clamp(0,1).numpy()
        tgt_img  = tgt.cpu().clamp(0,1).numpy()
        bic_img  = bic.cpu().clamp(0,1).numpy()
        for i, t, p, b in zip(info, tgt_img, pred_img, bic_img):
            tag = f'{i[0]}_{i[1]}'
            if tag not in outputs.keys():
                outputs[tag] = {
                    'vmin':float(i[2]), 'vmax':float(i[3]),
                    'x':[], 'pred':[], 'bic':[], 'tgt':[]
                }
            outputs[tag]['x'].append(float(i[4]))
            outputs[tag]['tgt'].append(t)
            outputs[tag]['pred'].append(p)
            outputs[tag]['bic'].append(b)
        return outputs
    
    def _parse_outputs(self, outputs):
        for v in outputs.values():
            o = np.argsort(v['x'])
            v['x'] = np.array(v['x'])[o].squeeze()
            v['tgt'] = np.stack(v['tgt'], axis=0)[o].squeeze()
            v['pred'] = np.stack(v['pred'], axis=0)[o].squeeze()
            v['bic'] = np.stack(v['bic'], axis=0)[o].squeeze()
        return outputs

class RBPNTrainer(BaseTrainer):
    def __init__(self, model, opt, residual=True, device=torch.device('cuda')):
        self.model = model
        self.opt   = opt
        self.residual = residual
        self.device = device
    
    def _iter_step(self, batch):
        inp, tgt, nbr, flow, bic, _ = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        nbr = nbr.to(self.device)
        flow = flow.to(self.device)
        bic = bic.to(self.device)

        pred = self.model(inp, nbr, flow)
        if self.residual:
            pred += bic

        return pred, tgt

    def _get_outputs(self, batch, output, outputs):
        _, tgt, _, _, bic, info = batch
        pred_img = output[0].cpu().clamp(0,1).numpy()
        tgt_img  = tgt.cpu().clamp(0,1).numpy()
        bic_img  = bic.cpu().clamp(0,1).numpy()
        for i, t, p, b in zip(info, tgt_img, pred_img, bic_img):
            tag = f'{i[0]}_{i[1]}'
            if tag not in outputs.keys():
                outputs[tag] = {
                    'vmin':float(i[2]), 'vmax':float(i[3]),
                    'x':[], 'pred':[], 'bic':[], 'tgt':[]
                }
            outputs[tag]['x'].append(float(i[4]))
            outputs[tag]['tgt'].append(t)
            outputs[tag]['pred'].append(p)
            outputs[tag]['bic'].append(b)
        return outputs
    
    def _parse_outputs(self, outputs):
        for v in outputs.values():
            o = np.argsort(v['x'])
            v['x'] = np.array(v['x'])[o].squeeze()
            v['tgt'] = np.stack(v['tgt'], axis=0)[o].squeeze()
            v['pred'] = np.stack(v['pred'], axis=0)[o].squeeze()
            v['bic'] = np.stack(v['bic'], axis=0)[o].squeeze()
        return outputs

class SSRNetTrainer(BaseTrainer):
    def __init__(self, model, opt, device=torch.device('cuda')):
        self.model = model
        self.opt   = opt
        self.device = device
    
    def _iter_step(self, batch):
        inp, tgt, _ = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        
        pred = self.model(inp)
        return pred, tgt

    def _get_outputs(self, batch, output, outputs):
        _, tgt, info = batch
        _tgt = tgt.detach().cpu().numpy()
        _pred = output[0].detach().cpu().numpy()
        tags = np.array([f'{i0}_{i1}' for i0, i1 in info[:,:2]])
        for tag in np.unique(tags):
            m = tags == tag
            _info = info[m].T
            if tag not in outputs.keys():
                outputs[tag] = {
                    'order':_info[4], 'vmin':_info[2,0], 'vmax':_info[3,0], 
                    'x':_info[5:,0], 'tgt':_tgt[m], 'pred':_pred[m]
                }
            else:
                outputs[tag]['order'] = np.hstack([outputs[tag]['order'], _info[4]])
                outputs[tag]['tgt'] = np.vstack([outputs[tag]['tgt'], _tgt[m]])
                outputs[tag]['pred'] = np.vstack([outputs[tag]['pred'], _pred[m]])
        return outputs
    
    def _parse_outputs(self, outputs):
        for v in outputs.values():
            o = np.argsort(v.pop('order'))
            m = v['x'] != 0
            v['x'] = v['x'][m].squeeze()
            v['tgt'] = v['tgt'].squeeze()[o][..., m]
            v['pred'] = v['pred'].squeeze()[o][..., m]
        return outputs

class SAETrainer(BaseTrainer):
    def __init__(self, model, opt, device=torch.device('cuda')):
        self.model = model
        self.opt = opt
        self.device = device

    def _iter_step(self, batch):
        inp, tgt, _ = batch
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        
        pred, latent = self.model(inp)
        return pred, tgt, latent

    def _get_outputs(self, batch, output, outputs):
        _, tgt, info = batch
        pred, _, latent = output
        _tgt = tgt.detach().cpu().numpy()
        _pred = pred.detach().cpu().numpy()
        _latent = latent.detach().cpu().numpy()
        tags = np.array([f'{i0}_{i1}' for i0, i1 in info[:,:2]])
        for tag in np.unique(tags):
            m = tags == tag
            _info = info[m].T
            if tag not in outputs.keys():
                outputs[tag] = {
                    'order':_info[4], 'vmin':_info[2,0], 'vmax':_info[3,0], 
                    'x':_info[5:,0], 'tgt':_tgt[m], 'pred':_pred[m], 'latent':_latent[m]
                }
            else:
                outputs[tag]['order'] = np.hstack([outputs[tag]['order'], _info[4]])
                outputs[tag]['tgt'] = np.vstack([outputs[tag]['tgt'], _tgt[m]])
                outputs[tag]['pred'] = np.vstack([outputs[tag]['pred'], _pred[m]])
                outputs[tag]['latent'] = np.vstack([outputs[tag]['latent'], _latent[m]])
        return outputs
    
    def _parse_outputs(self, outputs):
        for v in outputs.values():
            o = np.argsort(v.pop('order'))
            m = v['x'] != 0
            v['x'] = v['x'][m].squeeze()
            v['tgt'] = v['tgt'].squeeze()[o][..., m]
            v['pred'] = v['pred'].squeeze()[o][..., m]
            v['latent'] = v['latent'].squeeze()[o]
        return outputs