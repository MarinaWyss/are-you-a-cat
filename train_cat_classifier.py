import yaml
from cat_classifier import CatClassifier

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)

def main():
    args = configs['training']
    dnn_skill_tagging = CatClassifier(args)
    dnn_skill_tagging.construct()
    dnn_skill_tagging.fit()
    dnn_skill_tagging.save()


if __name__ == '__main__':
    main()
