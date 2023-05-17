OUTPUT_STR = " : lambda spec: ? if "
output_list = []

def find_value_in_dict(d, target_key):
    if target_key in d:
        return d[target_key]

    for k, v in d.items():
        if isinstance(v, dict):
            found_value = find_value_in_dict(v, target_key)
            if found_value is not None:
                return found_value
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    found_value = find_value_in_dict(item, target_key)
                    if found_value is not None:
                        return found_value

    return None

def update_value_in_dict(d, target_key, new_value):
    if target_key in d:
        d[target_key] = new_value
        return True

    for k, v in d.items():
        if isinstance(v, dict):
            updated = update_value_in_dict(v, target_key, new_value)
            if updated:
                return True
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    updated = update_value_in_dict(item, target_key, new_value)
                    if updated:
                        return True

    return False

with open("/home/kola/miniconda3/envs/autopytorch/lib/python3.8/site-packages/autoPyTorch-0.2.1-py3.8.egg/autoPyTorch/optimizer/OA_condition.txt") as file:
    for item in file:
        # print(item, end ="")
        item = item.replace('(', '').replace(')', '').replace('\n', '')  # remove any (, ) and space
        item_split = item.split('&&')
        i = 0
        output = ""
        for it in item_split:
            if i == 1:
                output = output + 'and '
            it_split = it.split('|')
            if i == 0:
                i = 1
                key = it_split[0].split(':')[-1]
                output = key + OUTPUT_STR
            a = it_split[1].replace(':', '.')
            a  = a[1:]
            output = output + 'spec.search_config.' + a

        output  = output + " else None"

        # print(output, '\n')
        output_list.append(output)

# print(output_list)

from autoPyTorch.optimizer.OA_cs import search_config

convert_config = search_config

for out in output_list:
    key = out.split(':')[0].replace(' ','')
    val = find_value_in_dict(convert_config, key)
    updated_str = out.replace(out.split(':')[0] + ':', '').replace("?", "val")
    # print(updated_str)

    my_lambda = eval(updated_str)
    update_value_in_dict(convert_config, key, my_lambda)
    # print(find_value_in_dict(convert_config, key))

print('Conversion end')
