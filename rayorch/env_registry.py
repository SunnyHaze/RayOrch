import json, copy
import yaml, requests
import subprocess
import os
from typing import Dict, List, Union, Tuple, Set, Optional
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import parse as parse_version
from packaging.version import Version

class NewRequirement(Requirement):

    def __init__(self, requirement_string: str, install_mode: str = 'conda'):
        super().__init__(requirement_string)
        self.install_mode = install_mode
        
class Environment:
    
    # TODO: 支持更多环境信息，例如环境变量、docker等，尽量与ray的runtime env对齐。目前仅支持conda和pip 的 环境管理、一对多、合并等。
    
    def __init__(self, input_env_dict: dict = None):
        
        if input_env_dict is not None:
            self.input_env_dict = input_env_dict
        self.run_with_default_env = input_env_dict.get("conda", []) == []

    # TODO: 重新实现环境依赖解析和管理。


class EnvironmentRegistry():

    def __init__(self, name):

        self._name = name
        self._env_map_requirements = {}
        self._env_map_ray_style = {}
        self._env_map_class = {}
        
    def _do_register(self, env_instance: Environment, name: str, env_input_dict : Optional[Dict] = None):
        
        self._env_map_class[name] = env_instance
        self._env_map_requirements[name] = env_instance.get_requires()
        self._env_map_ray_style[name] = env_instance.get_ray_runtime_env()


    def register(self, name: Optional[str] = None, env_input_dict : Optional[Dict] = None):
        
        if name is None:
            def deco(class_obj):
                env_instance = class_obj()
                self._do_register(env_instance, class_obj.__name__, env_instance.input_env_dict)
                
                return class_obj
    
            return deco
        
        self._do_register(Environment(env_input_dict), name, env_input_dict)
        
    def get_ray_style_env(self, name: str) -> dict:

        return self._env_map_ray_style.get(name)



EnvRegistry = EnvironmentRegistry("GlobalEnvRegistry")