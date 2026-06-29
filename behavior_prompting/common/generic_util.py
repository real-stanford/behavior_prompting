import os

def symlink_absolute(src, dest, **kwargs):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    os.symlink(src, dest, **kwargs)

def symlink_relative(src, dest, **kwargs):
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    relative_src_path = os.path.relpath(src, start=os.path.dirname(dest))
    os.symlink(relative_src_path, dest, **kwargs)
