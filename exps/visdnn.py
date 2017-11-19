#!/usr/bin/env python

# A python tool to generate a Image for visualizing a model

import argparse
import graphviz as gviz
import os
import sys


class NetBlock(object):
    """Represent a block in a network diagram"""

    def __init__(self, index, kind, name, color):
        self.index = index
        self.kind = kind
        self.name = name
        self.color = color


class NetDiagram(object):
    color_map = dict(
        input='white',
        linear='skyblue',
        pooling='salmon',
        concat='green',
        misc='aquamarine',
        ewise='aquamarine',
        loss='orange',
        bn='cyan',
        eltwise='seagreen')

    layer_kind_map = dict(
        Input='input',
        FC='linear',
        Module='bn',
        Convolution='linear',
        Conv='linear',
        FullyConnected='linear',
        InnerProduct='linear',
        Pooling='pooling',
        Concat='concat',
        Dropout='ewise',
        LRN='ewise',
        ReLU='ewise',
        Softmax='ewise',
        SoftmaxWithLoss='loss',
        Accuracy='loss',
        BN='bn',
        Sum='eltwise')

    def __init__(self, name):
        self._name = name
        self._blocks = []
        self._block_map = dict()
        self._links = []

    @property
    def name(self):
        return self._name

    def add_block(self, kind, name, color='royalblue'):

        if name in self._block_map:
            raise ValueError('Block name {} already existed'.format(name))

        idx = len(self._blocks)

        # add the block
        blk = NetBlock(idx, kind, name, color)
        self._blocks.append(blk)
        self._block_map[name] = blk

    def add_input(self, name):
        self.add_block('input', name, NetDiagram.color_map['input'])

    def add_layer(self, name, layertype):

        # get color
        kind = NetDiagram.layer_kind_map.get(layertype)
        if not kind:
            color = NetDiagram.color_map['misc']
        else:
            color = NetDiagram.color_map[kind]
        self.add_block('layer', name, color)

    def add_link(self, from_name, to_name):

        fi = self._block_map[from_name].index
        ti = self._block_map[to_name].index
        self._links.append((fi, ti))

    @property
    def size(self):
        return (self._xmin + self._xmax, self._ymin + self._ymax)

    def make_graph(self, format='pdf'):
        """Draw on an axis"""

        g = gviz.Digraph(format=format)
        g.attr('graph', rankdir='TB')
        for blk in self._blocks:
            g.node(blk.name, fillcolor=blk.color, style='filled', shape='rect')

        for (fi, ti) in self._links:
            f_id = self._blocks[fi].name
            t_id = self._blocks[ti].name
            g.edge(f_id, t_id)

        return g

    @staticmethod
    def build_from(model):
        """Build a network diagram from a DNN model

            Args:
                model:  a loaded model of class dnn.model
        """

        dg = NetDiagram(model['name'])

        # variable -> latest upstream
        upstream_map = dict()

        # add layer nodes
        for layer in model['layer_entries']:
            # add a new layer
            lid = layer['id']
            dg.add_layer(lid, layer['typename'])

            # add link from its dependent layers
            dep_layer_ids = []
            if 'bottom_ids' in layer:
                for vid in layer['bottom_ids']:
                    dep_id = upstream_map.get(vid)
                    if dep_id:
                        dep_layer_ids.append(dep_id)

            for dep_id in dep_layer_ids:
                dg.add_link(dep_id, lid)

            # make this layer the latest upstream of its outputs
            if 'top_ids' in layer:
                for out_vid in layer['top_ids']:
                    upstream_map[out_vid] = lid
            else:
                upstream_map[lid] = lid
        return dg


class StoreDict(argparse.Action):
    pairs = {}

    def __call__(self, parser, namespace, values, option_string=None):
        for kv in values.split(','):
            k, v = kv.split('=')
            self.pairs[k] = eval(v)
        setattr(namespace, self.dest, self.pairs)


class Struct(object):
    def __init__(self, entries):
        self.__dict__.update(entries)

    def __getitem__(self, item):
        return self.__dict__[item]


def dict2obj(d):
    return Struct(d)


import yaml, re

name_mapping = {
    'FullyConnected': 'InnerProduct'
}

expr_parser = re.compile('([^\s]+)\s?=\s?([^()]*)\((.*)\)')
shape_parser = re.compile('\s?([^()]*)\((.*)\)')


def parse_parrots_expr(expr):
    m = re.match(expr_parser, expr)
    out_str, op, in_str = m.groups()

    out_list = [x.strip() for x in out_str.strip().split(',')]
    in_list = [x.strip() for x in in_str.strip().split(',')]

    in_blob_list = [x for x in in_list if not x.startswith('@')]
    param_list = [x[1:] for x in in_list if x.startswith('@')]

    return out_list, op, in_blob_list, param_list


def parse_shape(expr):
    m = re.match(shape_parser, expr)
    dtype, shape_str = m.groups()
    shape = tuple([int(x.strip()) if x.strip() != '_' else 1 for x in shape_str.split(',')])
    return shape


if __name__ == '__main__':

    model_cfg = '''
    name: resnet.gl
    layer_entries: 
        - id: 'input(1,3,256,128)'
          typename: Input          
        - id: 'resnet50(1,2048,8,4)'
          typename: Module  
          bottom_ids:  [ 'input(1,3,256,128)' ]
        - id: avg(1,2048)
          typename: Pooling
          bottom_ids: [ 'resnet50(1,2048,8,4)']
        - id: 'fc(1,1024)'
          typename: FC 
          bottom_ids:  ['avg(1,2048)']
        - id: 'fc(1,128)'
          typename: FC 
          bottom_ids: ['fc(1,1024)']
        - id: 'norm'
          typename: Normalize
          bottom_ids:  ['fc(1,128)']   
    '''
    model_cfg = '''
       name: resnet.gl
       layer_entries:
           - expr: input=()
             shape: (160,3,256,128)  
           - expr: fea=ResNet50(input)
             shape: (160,2048,8,4) 
           - expr: avg1=Pooling(fea)
             shape: (160,2048) 
           - expr: global=FC(avg1) 
             shape: (160, 1024) 
             
           - expr: reduce_fea=Conv(fea) 
             shape: (160,128,8,4) 
             
           - expr: b1_conv=Conv(reduce_fea)
             shape: (160,1,8,4)
           - expr: b1_mask=Sigmoid(b1_conv)
             shape: (160,1,8,4) 
           - expr: b1_fea=Mul(b1_mask,reduce_fea) 
             shape: (160,128,8,4) 
           - expr: b1_local=Pooling(b1_fea) 
             shape: (160,128)  
           
           - expr: b2_conv=Conv(reduce_fea)
             shape: (160,1,8,4)
           - expr: b2_mask=Sigmoid(b2_conv)
             shape: (160,1,8,4) 
           - expr: b2_fea=Mul(b2_mask,reduce_fea) 
             shape: (160,128,8,4) 
           - expr: b2_local=Pooling(b2_fea) 
             shape: (160,128)  
             
           - expr: b8_conv=Conv(reduce_fea)
             shape: (160,1,8,4)
           - expr: b8_mask=Sigmoid(b8_conv)
             shape: (160,1,8,4) 
           - expr: b8_fea=Mul(b8_mask,reduce_fea) 
             shape: (160,128,8,4) 
           - expr: b8_local=Pooling(b8_fea) 
             shape: (160,128)  
                                       
                     
           - expr: concat=Concat(global,b1_local,b2_local,b8_local)
             shape: (160,2048 )
             
           - expr: fc_fea=FC(concat)
             shape: (160,128) 
             
              
       '''

    model_cfg = '''
           name: resnet.gl
           layer_entries:
               - expr: input=()
                 shape: (160,3,256,128)  
               - expr: fea=ResNet50(input)
                 shape: (160,2048,8,4) 
               - expr: avg1=Pooling(fea)
                 shape: (160,2048) 
               - expr: global=FC(avg1) 
                 shape: (160, 1024) 

               - expr: reduce_fea=Conv(fea) 
                 shape: (160,128,8,4) 

               - expr: b1_conv=Conv(reduce_fea)
                 shape: (160,1,8,4)
               - expr: b1_mask=Sigmoid(b1_conv)
                 shape: (160,1,8,4) 
               - expr: b1_fea=Mul(b1_mask,reduce_fea) 
                 shape: (160,128,8,4) 
               - expr: b1_local=Pooling(b1_fea) 
                 shape: (160,128)  

               - expr: b2_conv=Conv(reduce_fea)
                 shape: (160,1,8,4)
               - expr: b2_mask=Sigmoid(b2_conv)
                 shape: (160,1,8,4) 
               - expr: b2_fea=Mul(b2_mask,reduce_fea) 
                 shape: (160,128,8,4) 
               - expr: b2_local=Pooling(b2_fea) 
                 shape: (160,128)  

               - expr: b8_conv=Conv(reduce_fea)
                 shape: (160,1,8,4)
               - expr: b8_mask=Sigmoid(b8_conv)
                 shape: (160,1,8,4) 
               - expr: b8_fea=Mul(b8_mask,reduce_fea) 
                 shape: (160,128,8,4) 
               - expr: b8_local=Pooling(b8_fea) 
                 shape: (160,128)  

               - expr: concat=Concat(global,b1_local,b2_local,b8_local)
                 shape: (160,2048 )

               - expr: fc_fea=FC(concat)
                 shape: (160,128) 


           '''

    # parse command line arguments

    argparser = argparse.ArgumentParser(
        "visdnn", description="Deep Neural Network (DNN) visualization tool.")

    argparser.add_argument(
        '-o', '--output', help='destination path', default='./graph')
    argparser.add_argument(
        '--fmt', help='output format (e.g. pdf, png, svg, ...)', default='png')
    argparser.add_argument(
        '--modelargs',
        action=StoreDict,
        metavar='key1=val1,key2=val2...',
        default={})
    argparser.add_argument(
        '-V',
        '--view',
        help='open the generated graph image',
        action="store_true",
        default=True)
    args = argparser.parse_args()

    # build graph
    model = yaml.load(model_cfg)

    id2shape = {}
    for layer in model['layer_entries']:
        out_list, op, inp_list, _ = parse_parrots_expr(layer['expr'])

        layer['id'] = out_list[0]
        layer['top_ids'] = out_list
        if inp_list[0] != '':
            layer['bottom_ids'] = inp_list
        if op != '':
            layer['typename'] = op
        else:
            layer['typename'] = 'Input'
        id2shape[layer['id']] = parse_shape(layer['shape'])
    for layer in model['layer_entries']:
        layer['id'] = layer['id'] + str(id2shape[layer['id']])
        if 'bottom_ids' in layer:
            layer['bottom_ids'] = [id + str(id2shape[id]) for id in layer['bottom_ids']]
        if 'top_ids' in layer:
            layer['top_ids'] = [id + str(id2shape[id]) for id in layer['top_ids']]

    netdg = NetDiagram.build_from(model)
    g = netdg.make_graph(format=args.fmt)

    # output
    dst_path = args.output
    to_view = args.view
    print('Writing graph to {} ...'.format(dst_path))
    g.render(filename=dst_path, cleanup=True, view=to_view)
