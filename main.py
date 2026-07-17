import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import lognorm

# 1. Setup the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3) # Make room for sliders

# Generate x values (starting slightly above 0 to avoid log(0))
x = np.linspace(0.01, 10, 1000)

# Initial parameters
initial_mu = 0.0
initial_sigma = 0.5

# 2. Plot the initial log-normal distribution
# Note: scipy's lognorm takes 's' as sigma and 'scale' as exp(mu)
initial_pdf = lognorm.pdf(x, s=initial_sigma, scale=np.exp(initial_mu))
line, = ax.plot(x, initial_pdf, lw=2, color='blue')

ax.set_xlim(0, 10)
ax.set_ylim(0, 1.5)
ax.set_title('Log-Normal Distribution')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density')

# 3. Create the sliders
ax_mu = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_sigma = plt.axes([0.15, 0.08, 0.7, 0.03])

slider_mu = Slider(ax_mu, 'Mean ($\mu$)', -2.0, 2.0, valinit=initial_mu)
slider_sigma = Slider(ax_sigma, 'Std Dev ($\sigma$)', 0.1, 2.0, valinit=initial_sigma)

# 4. Define the update function for when sliders are moved
def update(val):
    mu = slider_mu.val
    sigma = slider_sigma.val
    
    # Calculate new PDF
    new_pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    
    # Update the line data
    line.set_ydata(new_pdf)
    
    # Dynamically adjust the y-axis so the peak doesn't go off-screen
    ax.set_ylim(0, max(new_pdf) * 1.1)
    
    # Redraw the canvas
    fig.canvas.draw_idle()

# 5. Connect the sliders to the update function
slider_mu.on_changed(update)
slider_sigma.on_changed(update)

plt.show()