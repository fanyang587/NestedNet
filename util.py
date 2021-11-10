import os
import numpy as np
import shutil

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def list_var_name(list_of_tensors):
    """输入一个Tensor列表，返回一个包含每个tensor名字的列表
    """
    return [var.name for var in list_of_tensors]


def get_var(list_of_tensors, prefix_name=None):
    """输入一个Tensor列表(可选变量名前缀)，返回[变量名],[变量]两个列表
    """
    if prefix_name is None:
        return list_var_name(list_of_tensors), list_of_tensors
    else:
        specific_tensor = []
        specific_tensor_name = []
        for var in list_of_tensors:
            if var.name.find(prefix_name)>0:
                specific_tensor.append(var)
                specific_tensor_name.append(var.name)
        return specific_tensor_name, specific_tensor

def split_var(list_of_tensors, prefix_name=None):
    """输入一个Tensor列表(可选变量名前缀)，返回[变量名],[变量]两个列表
    """
    if prefix_name is None:
        return list_var_name(list_of_tensors), list_of_tensors
    else:
        search_tensor = []
        model_tensor = []
        for var in list_of_tensors:
            print(var.name)
            if var.name.find(prefix_name)>0:
                search_tensor.append(var)
            else:
                model_tensor.append(var)
        return search_tensor, model_tensor
