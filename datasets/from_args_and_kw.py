from .datasets import (
    DEFAULT_DATA_DIR,
    AVAILABLE_DATASETS,
    simplified_get_train_test_dl,
    new_distributed_simplified_get_train_test_dl,
    get_separate_just_x_or_y_train_test_dl,
    get_separate_just_x_or_y_test_dl,
)

# new_distributed_get_train_valid_dl_from_args  (train, valid)
# simplified_get_train_valid_dl_from_args  (train, valid)
# get_separate_just_x_or_y_train_test_dl_from_args  (train, valid)
# get_separate_just_x_or_y_test_dl_from_args: (just the test dataloader)

# NOTE: **kw here are keywords for DataLoader.

###################################
# From args and key words.
###################################


def get_separate_just_x_or_y_train_test_dl_from_args(args, **kw):
    # TODO: according to ranks, for replicated stages.
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # Just:
    # HACK: avoid asking "is last partition?"  
    # FIXME: need to ask that for comined IntraLayer and Pipeline.
    just = 'x' if args.stage == 0 else 'y'

    # num_replicas=None, rank=None
    return get_separate_just_x_or_y_train_test_dl(
        args.dataset,
        args.bs_train,
        # TODO: change it to validation...
        args.bs_test,
        just,
        DATA_DIR=DATA_DIR,
        **kw)


def get_separate_just_x_or_y_test_dl_from_args(args, **kw):
    """ get just the test dataset.
    kw can have
    test_dataset_keywords=dict()
    to help with it
    """
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # Just:
    # HACK: avoid asking "is last partition?"
    just = 'x' if args.stage == 0 else 'y'

    # num_replicas=None, rank=None
    return get_separate_just_x_or_y_test_dl(args.dataset,
                                            args.bs_test,
                                            just,
                                            DATA_DIR=DATA_DIR,
                                            **kw)


def add_dataset_argument(parser, default='cifar10', required=False):
    parser.add_argument('--dataset',
                        default=default,
                        choices=list(AVAILABLE_DATASETS),
                        required=required)


def args_extractor1(args):
    """extracts:
        args.dataset, args.bs_train, args.bs_test, args.data_dir
    """
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR
    return dict(DATA_DIR=DATA_DIR,
                dataset=args.dataset,
                bs_train=args.bs_train,
                bs_test=args.bs_test)


def simplified_get_train_valid_dl_from_args(args,
                                            shuffle_train=True,
                                            verbose=True,
                                            **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    return simplified_get_train_test_dl(args.dataset,
                                        args.bs_train,
                                        args.bs_test,
                                        shuffle_train=shuffle_train,
                                        verbose=verbose,
                                        DATA_DIR=DATA_DIR,
                                        **kw)


def new_distributed_get_train_valid_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # num_replicas=None, rank=None
    return new_distributed_simplified_get_train_test_dl(args.dataset,
                                                        args.bs_train,
                                                        args.bs_test,
                                                        DATA_DIR=DATA_DIR,
                                                        **kw)