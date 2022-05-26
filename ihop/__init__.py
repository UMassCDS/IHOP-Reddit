import sys

if not "-m" in sys.argv:
    from . import clustering
    from . import community2vec
    from . import text_processing
    from . import visualizations
    from . import resources
