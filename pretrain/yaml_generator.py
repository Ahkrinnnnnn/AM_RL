import os
import yaml
import random

yamlName = "eagle_catch_nc"
yamlPath = os.path.dirname(__file__) + '/' + yamlName + ".yaml"


with open(yamlPath, "r") as file:
    data = yaml.safe_load(file)

for i in range(1000):
    yamlGenePath = os.path.dirname(__file__) + "/yaml/" + yamlName + '_' + str(i) + ".yaml"
    
    obj = [10 * (random.random()-0.5), 10 * (random.random()-0.5), random.random()]
    hover = [obj[0]*2, obj[1]*2, obj[2]+0.5]
    
    for stage in data['trajectory']['stages']:
        if stage['name'] == 'pre_grasp':
            stage['costs'][3]['position'] = obj
        if stage['name'] == 'grasp':
            stage['costs'][3]['position'] = obj
        if stage['name'] == 'hover':
            stage['costs'][3]['position'] = hover

    with open(yamlGenePath, "w") as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=True)

