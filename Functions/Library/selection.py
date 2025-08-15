#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse


def point_selection(data_folder='Data',
                    output_target_path='Targets/path.txt',
                    spacing=25,
                    delay=0):
    """
    Launch the `Functions.Library.track` module in a detached subprocess,
    ensuring the project root is on PYTHONPATH so that `Functions` can be found.
    Returns a dict with the subprocess PID and a message.
    """
    # 1) locate this script's directory and compute project root (two levels up)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # 2) build a module-style invocation using -m
    cmd = [
        sys.executable,
        '-m', 'Functions.Library.track',
        '--data_folder',        data_folder,
        '--output_target_path', output_target_path,
        '--spacing',            str(spacing),
        '--delay',              str(delay),
    ]

    # 3) inject project_root into PYTHONPATH so Python can locate the Functions package
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    # 4) spawn the subprocess, detached from the parent session
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        start_new_session=True
    )

    # Return only serializable data
    return {
        'pid': proc.pid,
        'message': f'Launched track module subprocess with PID={proc.pid}'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Spawn Functions.Library.track via module invocation'
    )
    parser.add_argument(
        '--data_folder', default='Data',
        help='Path to the data folder containing frame_img.png and metadata'
    )
    parser.add_argument(
        '--output_target_path', default='Targets/path.txt',
        help='File to write the planned path points'
    )
    parser.add_argument(
        '--spacing', type=float, default=35,
        help='Pixel spacing buffer for obstacle dilation'
    )
    parser.add_argument(
        '--delay', type=float, default=0,
        help='Delay in milliseconds between path planning steps'
    )
    args = parser.parse_args()

    result = point_selection(
        data_folder=args.data_folder,
        output_target_path=args.output_target_path,
        spacing=args.spacing
    )
    print(result['message'])
