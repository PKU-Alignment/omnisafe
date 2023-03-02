key = "xxx:logger_cfgs:use_wandb"
value = "True"


keys_list = key.split(":")
cur_dict = {
    keys_list[-1]: value
}

for key in reversed(keys_list[:-1]):
    cur_dict = {key: cur_dict}

print(cur_dict)
