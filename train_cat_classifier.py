import yaml
from cat_classifier import CatClassifier

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)

def main(args):
    cat_classifier = CatClassifier(args)
    cat_classifier.construct()
    cat_classifier.fit()
    cat_classifier.save()


if __name__ == '__main__':
    main(configs['training'])
