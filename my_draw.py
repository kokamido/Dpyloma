import time
import json
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as pl
from IPython import display
from matplotlib import pyplot as plt
from matplotlib.text import Text

class Settings:
	def init(metadir,datadir,fourierdir,picdir):
		Settings.meta_dir = metadir
		Settings.data_dir = datadir
		Settings.fourier_dir = fourierdir
		Settings.pic_dir = picdir

	def set_lineplot():
		plt.rcParams['figure.figsize'] = [20, 10]
		plt.rcParams['grid.color'] = 'gray'  
		plt.rcParams['axes.grid'] = True
		plt.rcParams['lines.linewidth'] = 5
		plt.rcParams['font.size'] = 28
		plt.rcParams['axes.labelweight'] = 'bold'

	def set_heatmap():
		plt.rcParams['font.size'] = 28
		plt.rcParams['figure.figsize'] = [20, 10]

class Draw:
	def heatmap(df,start,end,xticks,xticklabels,yticks,yticklabels,cmap,borders=False,transpose = False,save_path = ''):	   
		plt.clf()
		plt.cla()
		Settings.set_heatmap()
		df = pd.DataFrame(df.iloc[start:end])
		if transpose:
			df = df.transpose().iloc[::-1]
		if(borders!=False):
			pic = sns.heatmap(df,vmin=borders[0],vmax=borders[1],xticklabels=False,yticklabels=False,cmap=cmap,cbar_kws = dict(use_gridspec=False,location="top"))
		else:
			pic = sns.heatmap(df,xticklabels=False,yticklabels=False,cmap=cmap,cbar_kws = dict(use_gridspec=False,location="top"))
		pic.set_xticks(xticks)
		pic.set_xticklabels(xticklabels)
		pic.set_yticks(yticks)
		pic.set_yticklabels(np.flip(yticklabels))
		if save_path == '':
			plt.show()
		else:
			plt.savefig(save_path)
		del df
	def fourierSum(df, to, index, a0 = 0,names=None,width=3,color='b',show=True):
		data = (df[names] if names != None else df).iloc[index]
		Settings.set_lineplot()
		x = np.linspace(0,to,1000,dtype=np.float)
		y = np.zeros(1000,dtype=np.float)
		for name in (names if names != None else list(df)):
			if('Unnamed' in name or name == 'a0' or name=='b0'):
				continue
			n = float(''.join(name[1::]))
			y+=(np.cos if name[0]=='a' else np.sin)(x*n/to*2*3.14159265)*data[name]/3.14159265/2
		y+=a0
		plt.plot(x,y,lw = width,color=color)
		if show:
			plt.show()		

class Result:
	def __init__(self,_id):
		onlyfiles = [os.path.join(Settings.meta_dir, f) for f in os.listdir(Settings.meta_dir) if os.path.isfile(os.path.join(Settings.meta_dir, f))]
		path_to_meta = next(filter(lambda x: _id in x and '_meta' in x,onlyfiles))
		with open(path_to_meta,'r') as f:
			lines = f.readlines()
			self.meta = json.loads(lines[0])
			self.end_u = np.fromstring(lines[1][1:-2],sep=',',dtype=np.float)
			self.end_v = np.fromstring(lines[2][1:-2],sep=',',dtype=np.float)
			self.start_u = np.array(self.meta['InitStateU'],dtype = np.float)
			del self.meta['InitStateU']
			self.start_v = np.array(self.meta['InitStateV'],dtype = np.float)
			del self.meta['InitStateV']
			self.meta['TimeLineQuant']*=self.meta['TimeQuant']
			self.space_net = np.fromiter([i * self.meta['SpaceQuant'] for i in range(0,int(self.meta['SpaceRange']/self.meta['SpaceQuant']))],dtype=np.float)
	def check_data(self):
		if not hasattr(self,'data'):
			onlyfiles = [os.path.join(Settings.data_dir, f) for f in os.listdir(Settings.data_dir) if os.path.isfile(os.path.join(Settings.data_dir, f))]
			path_to_data = next(filter(lambda x: str(self.meta['Id']) in x and '_data' in x,onlyfiles))
			self.data = pd.read_csv(path_to_data,sep=';',dtype=np.float)
	def draw_end_u(self,color='g'):
		Settings.set_lineplot()
		sns.lineplot(self.space_net, self.end_u,color=color)
		plt.show()
	def draw_start_u(self,color='g'):
		Settings.set_lineplot()
		sns.lineplot(self.space_net, self.start_u,color=color)
		plt.show()
	def draw_end_v(self):
		Settings.set_lineplot()
		sns.lineplot(self.space_net, self.end_v)
		plt.show()
	def draw_start_v(self):
		set_lineplot()
		sns.lineplot(self.space_net, self.start_v)
		plt.show()
	def draw_u(self,index,color='b'):
		Settings.set_lineplot()
		self.check_data()
		kek = np.array(self.data.iloc()[int(index/self.meta['TimeLineQuant'])])
		sns.lineplot(self.space_net, kek,color=color)
		plt.show()
	def draw_minimax(self,start,end,x_freq,min_color='b',max_color='r'):
		self.check_data()
		start_scaled = max(0,int(start/self.meta['TimeLineQuant']))
		end_scaled = min(int(end/self.meta['TimeLineQuant']),len(self.data.index)-1)
		mins = np.zeros(end_scaled-start_scaled,dtype=np.float)
		maxs = np.zeros(end_scaled-start_scaled,dtype=np.float)
		for ind, row in self.data.iloc()[start_scaled:end_scaled].iterrows():
			mins[ind-start_scaled] = row.min()
			maxs[ind-start_scaled] = row.max()
		Settings.set_lineplot()
		plt.clf()
		x_freq_scaled = int(x_freq/self.meta['TimeLineQuant'])
		xticks = np.linspace(0,(end_scaled-start_scaled) - (end_scaled-start_scaled)%x_freq_scaled,x_freq+1,dtype=np.int)
		xticklabels = np.array([int(x*self.meta['TimeLineQuant']+start) for x in xticks],dtype=np.str)
		xticklabels[-1] = 't'
		yticks = np.linspace(0,self.meta['SpaceRange']/self.meta['SpaceQuant'],5,dtype=np.int)
		ytickslabels = np.linspace(0,self.meta['SpaceRange'],5,dtype=np.int)
		ytickslabels = np.array(ytickslabels,dtype=np.str)
		ytickslabels[0]=''
		ytickslabels[-1] = 'x'
		plt.plot(np.arange(start,end,self.meta['TimeLineQuant']),mins,color=min_color)
		plt.xticks = xticks
		plt.xticklabels = xticklabels
		plt.yticks = yticks
		plt.yticklabels = ytickslabels
		plt.plot(np.arange(start,end,self.meta['TimeLineQuant']),maxs,color=max_color)
		plt.xticks = xticks
		plt.xticklabels = xticklabels
		plt.yticks = yticks
		plt.yticklabels = ytickslabels
		plt.show()
	def draw_heatmap(self,start, end,x_freq,borders=False, cmap = 'rainbow', display = True):
		Settings.set_heatmap()
		plt.clf()
		self.check_data()
		start_scaled = max(0,int(start/self.meta['TimeLineQuant']))
		end_scaled = min(int(end/self.meta['TimeLineQuant']),len(self.data.index)-1)
		if display:
			savepath = ''
		else:
			savepath = Settings.pic_dir+'\\'+str(self.meta['Id'])+'_'+str(start)+'_'+str(end)+'.png'
		x_freq_scaled = int(x_freq/self.meta['TimeLineQuant'])
		xticks = np.linspace(0,(end_scaled-start_scaled) - (end_scaled-start_scaled)%x_freq_scaled,x_freq+1,dtype=np.int)
		xticklabels = np.array([int(x*self.meta['TimeLineQuant']+start) for x in xticks],dtype=np.str)
		xticklabels[-1] = 't'
		yticks = np.linspace(0,self.meta['SpaceRange']/self.meta['SpaceQuant'],5,dtype=np.int)
		ytickslabels = np.linspace(0,self.meta['SpaceRange'],5,dtype=np.int)
		ytickslabels = np.array(ytickslabels,dtype=np.str)
		ytickslabels[0]=''
		ytickslabels[-1] = 'x'
		Draw.heatmap(self.data,start_scaled,end_scaled,xticks,xticklabels,yticks,ytickslabels,cmap,borders,True,savepath)
	def draw_fourier(self,start,end,names):
		if not hasattr(self,'data_f'):
			onlyfiles = [os.path.join(Settings.fourier_dir, f) for f in os.listdir(Settings.fourier_dir) if os.path.isfile(os.path.join(Settings.fourier_dir, f))]
			path_to_data = next(filter(lambda x: str(self.meta['Id']) in x and '_Fourier' in x,onlyfiles))
			self.data_f = pd.read_csv(path_to_data,sep=' ',dtype=float)
		Settings.set_lineplot()
		start_scaled = int(start/self.meta['TimeLineQuant'])
		end_scaled = int(end/self.meta['TimeLineQuant'])
		start_scaled = max(0,int(start/self.meta['TimeLineQuant']))
		end_scaled = min(int(end/self.meta['TimeLineQuant']),len(self.data.index)-1)
		data = self.data_f[names].iloc[start_scaled:end_scaled]
		palette = sns.color_palette('tab20',len(names)) if len(names) > 3 else sns.color_palette(['r','b','g'],len(names))
		pic = data.plot.line(color = palette,legend=False)
		pic.legend()
		pic.set_xlim(start_scaled,end_scaled)
		pic.set_xticklabels(np.array([sas*self.meta['TimeLineQuant'] for sas in pic.get_xticks()]))
		xlbls = pic.get_xticklabels()
		xlbls[-1]='t'
		pic.set_xticklabels(xlbls)
		ylabels = np.array(list(map(lambda s: round(s,2), pic.get_yticks())),dtype=np.str)
		ylabels[-2] = 'C'
		ylabels[-1]=''
		pic.set_yticklabels(ylabels)
		plt.show()
	def draw_fourier_range(self,start,end,namestart,nameend):
		if not hasattr(self,'data_f'):
			onlyfiles = [os.path.join(Settings.fourier_dir, f) for f in os.listdir(Settings.fourier_dir) if os.path.isfile(os.path.join(Settings.fourier_dir, f))]
			path_to_data = next(filter(lambda x: str(self.meta['Id']) in x and '_Fourier' in x,onlyfiles))
			self.data_f = pd.read_csv(path_to_data,sep=' ',dtype=float)
		self.draw_fourier(start,end,list(self.data_f.loc[:,namestart:nameend]))
	
	
def get_kok(num):
	i = 0
	while num > 0:
		num = num // 10
		i+=1
	return i
