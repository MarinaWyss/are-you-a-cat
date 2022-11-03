from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import train_test_validation


def main():
    train_ds, test_ds = classification_dataset_from_directory(
        root='data', object_type='VisionData', image_extension='jpg')
    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)
    result.save_as_html('data_validation.html')


if __name__ == '__main__':
    main()
