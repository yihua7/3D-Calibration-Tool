import os
import imageio
import trimesh
import threading
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
 

def align_pcls(x, y):
	"""
	Input:
	x: (n, 3) point cloud
	y: (n, 3) point cloud
	Output:
	affine: (4, 4) affine matrix A, y = (A @ (x.T)).T
	"""

	# Convert to ndarray
	x = np.array(x)
	y = np.array(y)

	# Determine scaling factor
	radius_x = np.linalg.norm(x - x.mean(axis=0), axis=1).mean()
	radius_y = np.linalg.norm(y - y.mean(axis=0), axis=1).mean()
	scale = radius_y / (radius_x + 1e-7)

	# Determine rotation and translation
	x = scale * x
	center_x = x.mean(axis=0)
	center_y = y.mean(axis=0)
	x = x - center_x
	y = y - center_y
	h = x.T @ y
	u, _, v_T = np.linalg.svd(h)
	v = v_T.T
	rotation = v @ u.T
	translation = center_y - (rotation @ center_x[:, None])[:, 0]

	# Obtain affine matrix
	affine = np.eye(4)
	affine[:3, :3] = scale * rotation
	affine[:3, 3] = translation

	return affine


class FP_Recorder:
	def __init__(self, pl, name=None):
		self.fp_list = []
		self.pl = pl
		self.name = name if name is not None else 'fp_recorder'
		pv.global_theme.font.size = 10
		self.pl.add_text('\nPress D to delete last picked point \nPress S to save screenshot \nPress Q to quit', font_size=18)
		self.pl.add_key_event('d', self.callback_point_deleting)
		self.pl.add_key_event('s', self.callback_screenshot)
		# self.pl.add_key_event('q', self.callback_quit)
		self.screenshot_path = './' + self.name + '.png'
	
	def callback_cell_picking(self, cell):
		if cell is not None:
			point = cell.points.mean(axis=0)
			self.fp_list.append(point)
			self.pl.add_point_labels(point, [str(len(self.fp_list))], name=str(len(self.fp_list)), font_size=30)
			print(f'{len(self.fp_list)} points now. Add: ', point)
	
	def callback_point_picking(self, point):
		if point is not None:
			self.fp_list.append(point)
			self.pl.add_point_labels(point, [str(len(self.fp_list))], name=str(len(self.fp_list)), font_size=30)
			print(f'{len(self.fp_list)} points now. Add: ', point)
	
	def callback_point_deleting(self, *args):
		if len(self.fp_list) > 0:
			print('Delete a point: ', self.fp_list[-1])
			self.pl.remove_actor(str(len(self.fp_list)))
			self.fp_list = self.fp_list[:-1]
			print(f'{len(self.fp_list)} points now.')

	def callback_screenshot(self, *args):
		path = './' + self.name + '.png'
		self.pl.screenshot(filename=path)
		print('Saved screenshot in ', path)

	def callback_quit(self, *args):
		self.pl.screenshot(filename=self.screenshot_path)
		print('Saved screenshot in ', self.screenshot_path)
		self.pl.close()
		pv.close_all()


def mark_fp(mesh_path, pick_cell=False, name=None):
	trimesh_mesh = trimesh.load(mesh_path, process=False)
	colors = trimesh_mesh.visual.vertex_colors[:, :4]
	fp_list = []
	print('Reading mesh file...')
	mesh = pv.read(mesh_path)
	print('Plotting...')
	pl = pv.Plotter()
	recorder = FP_Recorder(pl=pl, name=name)
	if pick_cell:
		mesh["colors"] = colors[:, :3]
		pl.add_mesh(mesh, show_edges=True, scalars="colors", rgb=True)
		pl.enable_cell_picking(through=False, callback=recorder.callback_cell_picking, style='points')
	else:
		pl.add_points(mesh.points, render_points_as_spheres=True, scalars=colors, rgb=True)
		pl.enable_point_picking(callback=recorder.callback_point_picking, left_clicking=False, show_point=True)
	pl.show()
	fp_list = np.array(recorder.fp_list)
	print(fp_list, fp_list.shape)
	return fp_list, recorder.screenshot_path


def align_scenes(source_path, target_path, sv_path):
	fp_src = []
	while len(fp_src) == 0:
		print('Need to mark source feature points')
		fp_src, screenshot_path_src = mark_fp(source_path, name='src')
	print('Reading src screenshot')

	if os.path.exists(screenshot_path_src):
		screenshot_src = imageio.imread(screenshot_path_src)
		print('Showing src screenshot')
		plt.imshow(screenshot_src)
		plt.title('Source Feature Points as Reference')
		plt.show(block=False)
		plt.pause(1)
	
	fp_trg = []
	while len(fp_trg) != len(fp_src):
		print('Need to mark target feature points')
		fp_trg, _ = mark_fp(target_path, name='trg')

	print('Aligning...')
	affine = align_pcls(fp_src, fp_trg)
	mesh = trimesh.load_mesh(source_path)
	vertices = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], axis=-1)
	vertices = (affine @ vertices.T).T
	mesh.vertices = vertices[..., :3]
	mesh.export(sv_path)


if __name__ == '__main__':
	source_path = '../103_up_dj.ply'
	target_path = '../103_front_dj.ply'
	sv_path = '../103_up2front_dj.ply'
	align_scenes(source_path=source_path, target_path=target_path, sv_path=sv_path)
