

def parse_wandb_config(cfg:dict):

    parsed_cfg=dict()

    for key,value in cfg.items():
        keys = key.split('.')

        if len(keys) == 1:
            parsed_cfg[key] = value

        elif len(keys) == 2:
            try:
                parsed_cfg[keys[0].strip()][keys[1].strip()] = value
            except:
                parsed_cfg[keys[0].strip()] = {keys[1].strip():value}

            
        elif len(keys) == 3:
            try:
                parsed_cfg[keys[0].strip()][keys[1].strip()][keys[2].strip()] = value
            except:
                try:
                    parsed_cfg[keys[0].strip()][keys[1].strip()] = {keys[2].strip():value}
                except:
                    parsed_cfg[keys[0].strip()] = {keys[1].strip() : {keys[2].strip():value}}

    return parsed_cfg

print(parse_wandb_config(
    {
        'a':5,
        'network.pointer':5,
        'network.actor.size':5,
        'network.critic.yoyo':5,
        'training.critic.yoyo':5,
        'training.critic.yoyo':5,
    }
))