"""X 光异常区域基线 — deterministic numerical prototype."""
import argparse,json
from pathlib import Path
import numpy as np

def finite(a,ndim=None):
 a=np.asarray(a,dtype=float)
 if not np.isfinite(a).all() or not a.size or (ndim is not None and a.ndim!=ndim): raise ValueError('Invalid shape, empty or nonfinite input')
 return a
def process(image,reference,threshold=5.):
 image=finite(image,2);reference=finite(reference,2)
 if image.shape!=reference.shape or threshold<=0:raise ValueError('Shape mismatch or invalid threshold')
 residual=image-reference;center=np.median(residual);sigma=max(float(1.4826*np.median(np.abs(residual-center))),1e-8)
 score=np.abs(residual-center)/sigma
 return score>threshold,score

def demo():
 rng=np.random.default_rng(42);y,x=np.mgrid[:100,:140];reference=.3+.4*np.exp(-((x-70)**2+(y-50)**2)/1800);truth=((x-80)**2+(y-55)**2)<64;image=reference+rng.normal(0,.01,reference.shape)+truth*.2
 mask,score=process(image,reference);tp=int(np.sum(mask&truth));fp=int(np.sum(mask&~truth));recall=tp/int(truth.sum());assert recall>.99 and fp<10
 return None,{'Synthetic X-ray':image,'Anomaly mask':mask,'Robust residual score':score},{'synthetic_recall':recall,'false_positive_pixels':fp}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts'));args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 x,series,metrics=demo()
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 if x is None:
  fig,axes=plt.subplots(1,len(series),figsize=(14,4),layout='constrained');axes=np.atleast_1d(axes)
  for ax,(label,values) in zip(axes,series.items()):ax.imshow(values,cmap='viridis');ax.set_title(label);ax.set_axis_off()
 else:
  fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
  for label,values in series.items():ax.plot(x,values,label=label)
  ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Synthetic prototype validation — not field measurements')
 fig.savefig(args.output/'result.png',dpi=140);plt.close(fig)
 np.savez_compressed(args.output/'results.npz',**{str(i):v for i,v in enumerate(series.values())})
 (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
