def get_id(d):
    if 'split_index' in d:
        return 'split_{}'.format(d['split_index'])
    else:
        return 'leaf_{}'.format(d['leaf_index'])

def extract_tree(tree_list, eta=1.0, lmda=1.0):
    new_tree_list = []
    for idx, tree in enumerate(tree_list):
        nodes = tree['tree_structure'].copy()
        node_list = {}
        node_orders = []
        node_list, node_orders = extract_node(nodes=nodes, node_list=node_list, node_orders=node_orders)
        compute_node_logit(node_list, node_orders, eta=1.0, lmda=1.0)
        compute_node_logit_delta(node_list, node_orders)
        new_tree_list.append(node_list)
    return new_tree_list

def extract_node(nodes, parent=None, node_list={}, node_orders=[]):
    node = {}
    for k, v in nodes.items():
        if isinstance(v, dict):
            next
        else:
            node[k] = v

    node['is_leaf'] = not 'split_index' in node
    node['id'] = get_id(node)

    node_orders.append(node['id'])

    if parent:
        node['parent'] = parent['id']
    else:
        node['parent'] = None

    for k, v in nodes.items():
        if isinstance(v, dict):
            node[k] = get_id(v)
            extract_node(v, node, node_list, node_orders)
        else:
            next

    if node['is_leaf']:
        node['cover'] = node['leaf_count']
        node_list[node['id']] = node
    else:
        node['cover'] = node['internal_count']
        node_list[node['id']] = node

    return node_list, node_orders

def compute_node_logit(node_list, node_orders, eta, lmda):
    for k in reversed(node_orders):
        node = node_list[k]
        if node['is_leaf']:
            G = -1.* node['leaf_value'] * (node['leaf_count'] + lmda) / eta
        else:
            G = node_list[node['left_child']]['grad'] + node_list[node['right_child']]['grad']
        node_list[k]['grad'] = G
        node_list[k]['logit'] = -1. * G / (node['cover'] + lmda) * eta
    return node_list

def compute_node_logit_delta(node_list, node_orders):
    for k in reversed(node_orders):
        node = node_list[k]
        if node['parent'] is None:
            node['logit_delta'] = node['logit'] - .0
        else:
            node['logit_delta'] = node['logit'] - node_list[node['parent']]['logit']
    return node_list


def logit_contribution(tree_lst, leaf_lst):
    dist = {'intercept':0.0}
    for i, leaf in enumerate(leaf_lst):
        tree = tree_lst[i]
        node = tree['leaf_{}'.format(leaf)]
        parent_idx = node['parent']
        # print(node, parent_idx)
        while True:
            if parent_idx is None:
                dist['intercept'] += node['logit_delta']
                break
            else:
                parent = tree[parent_idx]
                feat = parent['split_feature']
                if not feat in dist:
                    dist[feat] = 0.0
                dist[feat] += node['logit_delta']
                node = tree[parent_idx]
                parent_idx = node['parent']
    return dist
