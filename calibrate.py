import os
import imageio
import trimesh
import threading
import numpy as np
import pyvista as pv
from multiprocessing import Pool
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
		self.pl.add_text('Press P to pick point \nPress D to delete last picked point \nPress X/Y to undo/redo. \nPress S to save screenshot \nPress Q to quit', font_size=12)
		self.pl.add_key_event('d', self.callback_point_deleting)
		self.pl.add_key_event('s', self.callback_screenshot)
		self.pl.add_key_event('x', self.callback_undo)
		self.pl.add_key_event('y', self.callback_redo)
		self.screenshot_path = './' + self.name + '.png'
		self.history = []
		self.history_pointer = 0
	
	def add_point(self, point):
		self.fp_list.append(point)
		self.pl.add_point_labels(point, [str(len(self.fp_list))], name=str(len(self.fp_list)), font_size=30)
		print(f'{self.name}: {len(self.fp_list)} points now. Add: ', point)
	
	def delete_last_point(self):
		print(f'{self.name}: Delete a point: ', self.fp_list[-1])
		self.pl.remove_actor(str(len(self.fp_list)))
		self.fp_list = self.fp_list[:-1]
		print(f'{self.name}: {len(self.fp_list)} points now.')
	
	def callback_cell_picking(self, cell):
		if cell is not None:
			point = cell.points.mean(axis=0)
			self.add_point(point)
			# Update history
			self.history = self.history[:self.history_pointer]
			self.history.append([1, point])
			self.history_pointer += 1
	
	def callback_point_picking(self, point):
		if point is not None:
			self.add_point(point)
			# Update history
			self.history = self.history[:self.history_pointer]
			self.history.append([1, point])
			self.history_pointer += 1
	
	def callback_point_deleting(self, *args):
		if len(self.fp_list) > 0:
			# Update history
			self.history = self.history[:self.history_pointer]
			self.history.append([0, self.fp_list[-1]])
			self.history_pointer += 1
			self.delete_last_point()
	
	def callback_undo(self, *args):
		if len(self.history) > 0 and self.history_pointer > 0:
			if self.history[self.history_pointer-1][0] == 1:
				# Last operation is picking point
				self.delete_last_point()
				self.history_pointer -= 1
			else:
				# Last operation is deleting point
				self.add_point(self.history[self.history_pointer-1][1])
				self.history_pointer -= 1
	
	def callback_redo(self, *args):
		if self.history_pointer < len(self.history):
			if self.history[self.history_pointer][0] == 1:
				# Next operation is picking point
				self.add_point(self.history[self.history_pointer][1])
				self.history_pointer += 1
			else:
				# Next operation is deleting point
				self.delete_last_point()
				self.history_pointer += 1

	def callback_screenshot(self, *args):
		path = './' + self.name + '.png'
		self.pl.screenshot(filename=path)
		print(f'{self.name}: Saved screenshot in ', path)


def mark_fp_single(mesh_path, pick_cell=False, name=None):
	trimesh_mesh = trimesh.load(mesh_path, process=False)
	if hasattr(trimesh_mesh.visual, 'vertex_colors'):
		colors = trimesh_mesh.visual.vertex_colors[:, :4]
	else:
		colors = trimesh_mesh.visual.material.to_color(trimesh_mesh.visual.uv)
	fp_list = []
	print('Reading mesh file...')
	mesh = pv.PolyData(trimesh_mesh.vertices, np.concatenate([3*np.ones_like(trimesh_mesh.faces[:, :1]), trimesh_mesh.faces], axis=-1).reshape([-1, 4]))
	print('Plotting...')
	pl = pv.Plotter(title=name)
	recorder = FP_Recorder(pl=pl, name=name)
	if pick_cell:
		mesh["colors"] = colors[:, :3]
		pl.add_mesh(mesh, show_edges=True, scalars="colors", rgb=True)
		pl.enable_cell_picking(through=False, callback=recorder.callback_cell_picking, style='points', show_message=False)
	else:
		mesh["colors"] = colors[:, :3]
		pl.add_mesh(mesh, show_edges=True, scalars="colors", rgb=True)
		pl.add_points(mesh.points, render_points_as_spheres=True, scalars=colors, rgb=True)
		pl.enable_point_picking(callback=recorder.callback_point_picking, left_clicking=False, show_point=True, show_message=False)
	pl.show()
	fp_list = np.array(recorder.fp_list)
	print(fp_list, fp_list.shape)
	return fp_list, recorder.screenshot_path


def align_scenes_mark_1by1(source_path, target_path, sv_path):
	# Mark source
	fp_src = []
	while len(fp_src) == 0:
		print('Need to mark source feature points')
		fp_src, screenshot_path_src = mark_fp_single(source_path, name='Source')
	print('Reading src screenshot')

	# Plot source marks
	if os.path.exists(screenshot_path_src):
		screenshot_src = imageio.imread(screenshot_path_src)
		print('Showing src screenshot')
		plt.imshow(screenshot_src)
		plt.title('Source Feature Points as Reference')
		plt.show(block=False)
		plt.pause(1)
	
	# Mark target
	fp_trg = []
	while len(fp_trg) != len(fp_src):
		print('Need to mark target feature points')
		fp_trg, _ = mark_fp_single(target_path, name='Target')

	# Aligning source to target
	print('Aligning...')
	affine = align_pcls(fp_src, fp_trg)
	mesh = trimesh.load_mesh(source_path, process=False)
	vertices = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], axis=-1)
	vertices = (affine @ vertices.T).T
	mesh.vertices = vertices[..., :3]

	# Set material
	mesh.visual.material.ambient = 255 * np.ones_like(mesh.visual.material.ambient)
	mesh.visual.material.diffuse = 255 * np.ones_like(mesh.visual.material.diffuse)
	mesh.visual.material.specular = 255 * np.ones_like(mesh.visual.material.specular)
	mesh.export(sv_path)
	print('Saved in ', sv_path)


def mark_fp_double(source_path, target_path):
	with Pool(2) as p:
		result_src, result_trg = p.starmap(mark_fp_single, [(source_path, None, 'Source'), (target_path, None, 'Target')])
	fp_list_src, _ = result_src
	fp_list_trg, _ = result_trg
	return fp_list_src, fp_list_trg


def align_scenes(source_path, target_path, sv_path):
	# Mark source
	fp_src, fp_trg = [], []
	while len(fp_src) != len(fp_trg) or len(fp_src) < 1:
		print('Need to annotate mark points.')
		fp_src, fp_trg = mark_fp_double(source_path, target_path)

	# Aligning source to target
	print('Aligning...')
	affine = align_pcls(fp_src, fp_trg)
	mesh = trimesh.load_mesh(source_path, process=False)
	vertices = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[:, :1])], axis=-1)
	vertices = (affine @ vertices.T).T
	mesh.vertices = vertices[..., :3]

	# Set material
	mesh.visual.material.ambient = 255 * np.ones_like(mesh.visual.material.ambient)
	mesh.visual.material.diffuse = 255 * np.ones_like(mesh.visual.material.diffuse)
	mesh.visual.material.specular = 255 * np.ones_like(mesh.visual.material.specular)
	mesh.export(sv_path)
	print('Saved in ', sv_path)


if __name__ == '__main__':
	source_path = '../eva/103A.ply'
	target_path = '../eva/103A3.align.ply'
	sv_path = '../eva/103A_.obj'
	align_scenes(source_path=source_path, target_path=target_path, sv_path=sv_path)
