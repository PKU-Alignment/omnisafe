import os
import omnisafe

# Just fill your experiment's log directory in here.
LOG_DIR = ""

if __name__ == '__main__':
    evaluator = omnisafe.Evaluator()
    for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved_model(save_dir=LOG_DIR, 
                                    model_name=item.name)
            evaluator.render(num_episode=10, camera_name='track', width=256, height=256)