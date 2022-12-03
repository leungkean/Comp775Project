# Register Datasets
import afa.datasets.retina
import afa.datasets.gastro

# Register Environment
import afa.environments


# The below code is a workaround to disable a warning message that is coming from
# TensorFlow Probability and dm-tree. Eventually, this code can probably be removed,
# once the warning has been addressed in those packages.
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
