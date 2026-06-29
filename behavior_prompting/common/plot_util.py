import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from textwrap import wrap
from PIL import Image

def plot_progressions(out_path, pred_progressions, gt_progressions, titles):
    """
    Plots multiple progressions in a square grid of subplots with a single legend.

    Parameters:
    - out_path: Path to save the output plot.
    - pred_progressions: List of predicted progressions.
    - gt_progressions: List of ground truth progressions.
    - titles: Optional list of titles corresponding to each progression.
    """
    num_plots = len(pred_progressions)
    
    # Determine grid size
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()  # Flatten axes for easier indexing

    for i in range(len(axes)):
        if i < num_plots:
            timesteps = np.arange(len(pred_progressions[i]))
            axes[i].plot(timesteps, pred_progressions[i], label='Predicted', color='blue')
            axes[i].plot(timesteps, gt_progressions[i], label='Ground Truth', color='orange')
            axes[i].set_xlabel('Timesteps')
            axes[i].set_ylabel('Progression')
            
            title = titles[i]
            title_wrapped = "\n".join(wrap(title, 30))
            axes[i].set_title(title_wrapped, fontsize=8)
        else:
            # Hide any extra axes
            axes[i].axis('off')

    # Create a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()  # Get legend handles/labels from the first subplot
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, frameon=False)
    
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_progression_and_frame(image, pred_progression, gt_progression=None, num_timesteps=None, title=""):
    """
    Plots a progression over an image.

    Parameters:
    - image: numpy image to use as background of plot.
    - pred_progression: Predicted progressions for single task.
    - gt_progression: Ground truth progressions for single task.
    - title: Optional title for the plot.
    """
    if num_timesteps is None:
        num_timesteps = len(pred_progression)

    fig, ax = plt.subplots(figsize=(4,4))
    canvas = fig.canvas
    timesteps = np.arange(len(pred_progression))
    ax.plot(timesteps, pred_progression)
    if gt_progression is not None:
        ax.plot(timesteps, gt_progression)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Progression')
    ax.set_xlim(0, num_timesteps)
    ax.set_ylim(0, 1)
    title_wrapped = "\n".join(wrap(title, 30))
    ax.set_title(title_wrapped)
    plt.subplots_adjust(top=0.8, left=0.15) # Make room for title and y axis label
    
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_argb(), dtype='uint8')  # (H * W * 4,)
    plot_image = image_flat.reshape(*reversed(canvas.get_width_height()), 4)  # (H, W, 4)
    plot_image = plot_image[:, :, 1:] # remove alpha channel

    # resize image to match plot_image height
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    image = Image.fromarray(image)
    image = image.resize((image.width * plot_image.shape[0] // image.height, plot_image.shape[0]))

    # combine image and plot_image
    final_image = np.zeros((plot_image.shape[0], image.width + plot_image.shape[1], 3), dtype=np.uint8)
    final_image[:, :image.width] = np.array(image)
    final_image[:, image.width:] = plot_image

    plt.close()

    return final_image

def plot_gripper_width(gripper_widths, gripper_detection_types, frame_i, out_width=480, out_height=360):
    """
    Plots a gripper width in a chart.

    Parameters:
    - gripper_widths: List of gripper widths.
    - gripper_detection_types: detection types for each gripper width (see `gripper_util.py` for description of values).
    - frame_i: Index of the current frame.
    - title: Optional title for the plot.
    """
    seq_len = len(gripper_widths)
    gripper_widths = gripper_widths[:frame_i] * 100  # Convert to cm
    gripper_detection_types = gripper_detection_types[:frame_i]

    dpi = 100
    fig, ax = plt.subplots(figsize=(out_width / dpi,out_height / dpi), dpi=dpi)
    canvas = fig.canvas
    timesteps = np.arange(len(gripper_widths))
    ax.plot(timesteps, gripper_widths, zorder=0)

    # Highlight gripper detection types
    needs_legend = False
    types = [(1, 'Both tags', 'r'), (2, 'Left only', 'g'), (3, 'Right only', 'purple')]
    for i, (type_val, label, color) in enumerate(types):
        detected = np.where(gripper_detection_types == type_val)[0]
        if len(detected) > 0:
            ax.scatter(detected, gripper_widths[detected], label=label, color=color, zorder=i+1)
            needs_legend = True


    ax.set_xlabel('Frame index', labelpad=-1)
    ax.set_ylabel('Gripper width (cm)')
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 10)
    if needs_legend:
        ax.legend()
    
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_argb(), dtype='uint8')  # (H * W * 4,)
    plot_image = image_flat.reshape(*reversed(canvas.get_width_height()), 4)  # (H, W, 4)
    plot_image = plot_image[:, :, 1:] # remove alpha channel

    plt.close()

    return plot_image

if __name__ == '__main__':
    # pred_progression = [0, 0.1, 0.3, 0.7, 1]
    # gt_progression = [0, 0.15, 0.25, 0.8, 1]
    # preds = [pred_progression] * 3
    # gts = [gt_progression] * 3
    # titles = ['really long text title that abcdefghijklmnopqrstuvwxyzabcdefghijklmnop', '2', '3']
    # plot_progressions('tmp_plot.png', preds, gts, titles)

    # im = plot_progression_and_frame(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8), pred_progression, gt_progression, len(pred_progression), titles[0])
    # Image.fromarray(im).save('tmp_plot_over_image.png')


    im = plot_gripper_width(np.random.rand(100)*0.08, np.random.randint(0, 4, (100)), 50)
    Image.fromarray(im).save('tmp_gripper_width.png')
    