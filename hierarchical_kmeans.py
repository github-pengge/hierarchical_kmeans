# -*- encoding: utf-8 -*-
import numpy as np
from scipy.cluster.vq import *
import pickle


class tree(object):
	def __init__(self, name, data=None, additional_data=None, children=None):
		self.name = name
		self.data = data
		self.additional_data = additional_data
		if children is None:
			self.children = []
		else:
			self.children = children

	def set_data(self, data):
		self.data = data

	def get_data(self):
		return self.data

	def set_additional_data(self, additional_data):
		self.additional_data = additional_data

	def get_additional_data(self):
		return self.additional_data

	def get_name(self):
		return self.name

	def set_children(self, children):
		self.children = children

	def add_child(self, child):
		self.children.append(child)

	def add_children(self, children):
		self.children.extend(children)

	def get_children_number(self):
		return len(self.children)

	def gather_data(self, gather_additional_data=False):
		'''Gather data from its children, depth=1'''
		if gather_additional_data:
			return [child.get_additional_data() for child in self.children]
		else:
			return [child.get_data() for child in self.children]

	def find(self, data, find_additional_data=False):
		if find_additional_data:
			if self.additional_data == data:
				return self
			else:
				for child in self.children:
					res = child.find(data, find_additional_data)
					if res != -1:
						return res
				return -1
		else:
			if self.data == data:
				return self
			else:
				for child in self.children:
					res = child.find(data, find_additional_data)
					if res != -1:
						return res
				return -1

	def is_leaf_node(self):
		return len(self.children) == 0

	def gather_leaves(self, output):
		assert type(output) == list, 'output must be a list.'
		if self.is_leaf_node():
			output.append(self)
			return
		for child in self.children:
			child.gather_leaves(output)

	def gather_data_from_leaves(self, output, gather_additional_data=False):
		assert type(output) == list, 'output must be a list.'
		if gather_additional_data:
			if self.is_leaf_node():
				output.append(self.additional_data)
				return
			for child in self.children:
				child.gather_data_from_leaves(output, gather_additional_data)
		else:
			if self.is_leaf_node():
				output.append(self.data)
				return
			for child in self.children:
				child.gather_data_from_leaves(output, gather_additional_data)

	def __str__(self):
		return str(self.name)

	def __repr__(self):
		return str(self)

	def __iter__(self):
		return iter(self.children)

	def __hash__(self):
		return hash(str(self.name))


class hierarchical_kmeans(object):
	def __init__(self, clusters):
		self.clusters = clusters  # tree type variable
		self.root = tree('root')
		self.total_k = 0

	def cluster(self, data, only_store_id=True, iteration=1):
		data = np.asarray(data)
		self.only_store_id = only_store_id
		cluster = self.clusters
		current_node = self.root
		idx = np.arange(data.shape[0])
		print('Doing hierarchical kmeans...')
		self._cluster(data, idx, current_node, cluster, iteration=iteration, only_store_id=only_store_id)
		print('Done! Actual total number of clusters is %d.' % self.total_k)

	def _cluster(self, data, idx, current_node, cluster_node, iteration=1, only_store_id=True):
		if cluster_node.is_leaf_node():
			self.total_k += 1
			if only_store_id:
				current_node.set_additional_data(idx)
			else:
				current_node.set_additional_data(data[idx])
			return

		codebook, _ = kmeans(data[idx], cluster_node.get_children_number(), iter=iteration)
		ids, _ = vq(data[idx], codebook)

		if codebook.shape[0] == 1:  # kmeans only find one cluster
			current_node.set_data(codebook[0])
			if only_store_id:
				current_node.set_additional_data(idx)
			else:
				current_node.set_additional_data(data[idx])
			return

		for i, c in enumerate(cluster_node):
			if i >= codebook.shape[0]:  # cluster number smaller than the number of next branch
				break
			idx_i = idx[np.nonzero(ids == i)[0]]
			branch = tree(c.get_name(), codebook[i])
			current_node.add_child(branch)
			self._cluster(data, idx_i, branch, c, iteration=iteration, only_store_id=only_store_id)

	def save(self, file_name):
		with open(file_name, 'wb') as f:
			pickle.dump([self.clusters, self.root, self.total_k], f)

	@staticmethod
	def load(file_name):
		hk = hierarchical_kmeans(None)
		with open(file_name, 'rb') as f:
			[hk.clusters, hk.root, hk.total_k] = pickle.load(f)
		return hk

	def find_cluster(self, data, max_depth=-1):
		data = np.asarray(data)
		if len(data.shape) == 1:
			return self._find_cluster([data], self.root, 0, max_depth)
		return [self._find_cluster([d], self.root, 0, max_depth) for d in data]

	def _find_cluster(self, data, current_node, current_depth, max_depth=-1):
		if current_node.is_leaf_node() or (max_depth >= current_depth and max_depth > 0):
			return current_node
		codebook = current_node.gather_data()
		id, _ = vq(data, codebook)
		current_depth += 1
		branch = current_node.children[id[0]]
		# print('id: %d' % id[0], 'branch: %s' % branch)
		return self._find_cluster(data, branch, current_depth, max_depth)
		

def test_tree():
	print('-' * 20)
	t = tree('root', 0)
	d = []
	x = tree('level1,node@1', 1)
	y = tree('level1,node@2', 2)
	z = tree('level1,node@3', 3)
	# t.add_children([x,y,z])
	t.add_child(x)
	t.add_child(y)
	t.add_child(z)
	x.add_child(tree('level2,child@1,node@1', 4))
	d.append(4)
	x.add_child(tree('level2,child@1,node@2', 5))
	d.append(5)
	y.add_child(tree('level2,child@2,node@1', 6))
	d.append(6)
	d.append(z.get_data())
	print(t.children)
	print(t.children[0].children)
	print(t.find(6))
	print(t.is_leaf_node())  # False
	print(t.gather_data())
	data = []
	t.gather_data_from_leaves(data)
	assert data == d
	for i, child in enumerate(t):
		print(i, child)
	leaves = []
	t.gather_leaves(leaves)
	print(leaves)
	print('Passed.')


def test_hk():
	print('-' * 20)
	data = np.random.randn(2000, 10)
	clusters = tree('root', 0)
	x = tree('l1-c1@1', 1)
	y = tree('l1-c2@2', 2)
	z = tree('l1-c3@3', 3)
	clusters.add_child(x)
	clusters.add_child(y)
	clusters.add_child(z)
	z = tree('l2-x@1', 3)
	w = tree('l2-x@2', 2)
	p1 = tree('l2-y@1', 2)
	p2 = tree('l2-y@2', 2)
	p3 = tree('l2-y@3', 2)
	x.add_children([z,w])
	y.add_children([p1,p2,p3])
	hk = hierarchical_kmeans(clusters)
	hk.cluster(data)
	hk.save('hk.pkl')
	print(clusters.children[0].children)
	print(hk.root.children[0].children)
	hk = hierarchical_kmeans.load('hk.pkl')
	print(hk.root.children[0].children, len(hk.root.children[0].children[0].additional_data))
	print('data[0] clustered to node %s' % hk.find_cluster(data[0]))
	print(0 in hk.root.children[0].children[0].get_additional_data())  # when hk.find_cluster(data[0])=='l2-x@1', this must be True.
	print('Passed? Run several times and check last output. When last but two output is "l2-x@1", it must be True.')


if __name__ == '__main__':
	test_tree()
	# test_hk()
