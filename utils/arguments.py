from simpletransformers.config.model_args import ClassificationArgs


def get_arguments():
    model_args = ClassificationArgs()
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.save_best_model = False
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 150
    model_args.evaluate_during_training_verbose = True
    model_args.overwrite_output_dir = True
    model_args.logging_steps = 150
    model_args.save_steps = 150
    model_args.max_seq_length = 512
    model_args.wandb_project = 'arabic-readability-assessment'
    return model_args
