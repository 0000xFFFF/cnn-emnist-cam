import os

models_dir = "models"


def makedir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def d_models(model_name):
    makedir(models_dir)
    return os.path.join(models_dir, model_name)


def tf_load_model(model):
    from tensorflow.keras.models import load_model
    return load_model(d_models(model))


def selectmodel(findname=""):
    i = 0
    models = {}
    for file in os.listdir(models_dir):
        if not file.endswith(".keras"):
            continue
        i += 1
        models[i] = file

    if findname:
        for k, f in models.items():
            if findname in f:
                return tf_load_model(f)

    if len(models) == 0:
        print("no models found")
        exit()

    model_str = models[1]

    if len(models) > 1:
        print("Please select a model:")

        for k, f in models.items():
            print(f"\t {k}. {f}")

        try:
            selection = int(input("num: "))
        except:
            exit()

        model_str = models[selection]

    print(f"selected model: {model_str}")

    return tf_load_model(model_str)
