
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# For reproducibility
np.random.seed(42)

# Using the 'df' created in the previous cell (xq1lGRWexF51)
# If running this cell independently, ensure 'df' is loaded or created.

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
X_train = train_df['Experience'].values.reshape(-1,1)
y_train = train_df['Salary'].values.reshape(-1,1)
X_test = test_df['Experience'].values.reshape(-1,1)
y_test = test_df['Salary'].values.reshape(-1,1)

# Add bias term (column of ones) for linear model y = w0 + w1*x
def add_bias(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

# Loss functions and gradients
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Gradients for MSE
def mse_gradient(X_b, y_true, y_pred):
    # returns gradient vector shape (2,1)
    m = X_b.shape[0]
    grad = -2/m * X_b.T.dot(y_true - y_pred)
    return grad

# Subgradient for MAE (use sign)
def mae_subgradient(X_b, y_true, y_pred):
    m = X_b.shape[0]
    sign = np.sign(y_true - y_pred)
    # For exact zeros, sign=0 which is a valid subgradient
    grad = -1/m * X_b.T.dot(sign)
    return grad

# Gradient Descent implementation
def gradient_descent(X_b, y, loss='mse', lr=0.01, n_iters=1000, w_init=None):
    n_features = X_b.shape[1]
    if w_init is None:
        w = np.zeros((n_features,1))
    else:
        w = w_init.copy()
    history = {'loss': [] , 'w': []}
    for i in range(n_iters):
        y_pred = X_b.dot(w)
        if loss=='mse':
            loss_val = mse_loss(y, y_pred)
            grad = mse_gradient(X_b, y, y_pred)
        elif loss=='mae':
            loss_val = mae_loss(y, y_pred)
            grad = mae_subgradient(X_b, y, y_pred)
        else:
            raise ValueError('Unsupported loss')
        w = w - lr*grad
        history['loss'].append(loss_val)
        history['w'].append(w.copy())
    return w, history

# Utility to compute predictions and metrics
def evaluate_model(w, X_b, y_true):
    y_pred = X_b.dot(w)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}

# 5) Experiment 1: Learning Rate Analysis (fix loss=MSE)
learning_rates = [0.00005, 0.0001, 0.0005, 0.001] # Adjusted learning rates
n_iters = 1000
results_exp1 = {}
for lr in learning_rates:
    w, history = gradient_descent(X_train_b, y_train, loss='mse', lr=lr, n_iters=n_iters)
    eval_train = evaluate_model(w, X_train_b, y_train)
    eval_test = evaluate_model(w, X_test_b, y_test)
    results_exp1[lr] = {
        'w': w,
        'history': history,
        'train_metrics': eval_train,
        'test_metrics': eval_test
    }
    print(f"LR={lr}: train_mse={eval_train['mse']:.2f}, test_mse={eval_test['mse']:.2f}")

# Find best lr by test MSE
best_lr = min(results_exp1.keys(), key=lambda lr: results_exp1[lr]['test_metrics']['mse'])
print(f"Best learning rate by test MSE: {best_lr}")

# Plot loss vs iterations for different learning rates
plt.figure(figsize=(8,6))
for lr in learning_rates:
    loss_vals = results_exp1[lr]['history']['loss']
    plt.plot(loss_vals, label=f'lr={lr}')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Training MSE (log scale)')
plt.title('Loss convergence for different learning rates (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('exp1_lr_convergence.png')
plt.show()

# 6) Experiment 2: Loss Function Comparison (fix lr to best_lr)
losses = ['mse','mae']
results_exp2 = {}
for loss in losses:
    w, history = gradient_descent(X_train_b, y_train, loss=loss, lr=best_lr, n_iters=n_iters)
    eval_train = evaluate_model(w, X_train_b, y_train)
    eval_test = evaluate_model(w, X_test_b, y_test)
    results_exp2[loss] = {
        'w': w,
        'history': history,
        'train_metrics': eval_train,
        'test_metrics': eval_test
    }
    print(f"Loss={loss}: train_mse={eval_train['mse']:.2f}, test_mse={eval_test['mse']:.2f}, train_mae={eval_train['mae']:.2f}")

# Plot loss vs iterations for different loss functions
plt.figure(figsize=(8,6))
for loss in losses:
    loss_vals = results_exp2[loss]['history']['loss']
    plt.plot(loss_vals, label=loss)
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.title(f'Loss convergence (lr={best_lr})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('exp2_loss_comparison.png')
plt.show()

# 7) Experiment 3: Combined Analysis (learning rates: [0.001,0.01], losses ['mse','mae'])
grid_lrs = [0.00005, 0.0001, 0.0005] # Adjusted grid learning rates
grid_losses = ['mse','mae']
results_exp3 = {}
for lr in grid_lrs:
    for loss in grid_losses:
        w, history = gradient_descent(X_train_b, y_train, loss=loss, lr=lr, n_iters=n_iters)
        eval_train = evaluate_model(w, X_train_b, y_train)
        eval_test = evaluate_model(w, X_test_b, y_test)
        results_exp3[(lr,loss)] = {'w': w, 'history': history, 'train_metrics': eval_train, 'test_metrics': eval_test}
        print(f"lr={lr}, loss={loss}: test_mse={eval_test['mse']:.2f}, test_mae={eval_test['mae']:.2f}")

# Visualizations: Model Fit for best and worst configurations by test MSE
# Determine best and worst in grid by test MSE
best_config = min(results_exp3.keys(), key=lambda k: results_exp3[k]['test_metrics']['mse'])
worst_config = max(results_exp3.keys(), key=lambda k: results_exp3[k]['test_metrics']['mse'])
print(f"Best config: {best_config}, Worst config: {worst_config}")

# Scatter plot with regression lines
plt.figure(figsize=(8,6))
plt.scatter(X_test.flatten(), y_test.flatten(), alpha=0.6, label='Test data')
# best line
w_best = results_exp3[best_config]['w']
x_line = np.linspace(X_test.min(), X_test.max(), 100)
x_line_b = add_bias(x_line.reshape(-1,1))
y_best = x_line_b.dot(w_best)
plt.plot(x_line, y_best, label=f'Best {best_config}', linewidth=2)
# worst line
w_worst = results_exp3[worst_config]['w']
y_worst = x_line_b.dot(w_worst)
plt.plot(x_line, y_worst, label=f'Worst {worst_config}', linewidth=2)
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.title('Model fit: Best vs Worst configurations (on test set)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('model_fit_best_worst.png')
plt.show()

# Optional: 3D plot of parameter space and gradient descent path for MSE with best lr
from mpl_toolkits.mplot3d import Axes3D
# We'll sample a grid around learned parameters
w_history = results_exp3[best_config]['history']['w']
w_hist_arr = np.hstack([w.reshape(2,1) for w in w_history])

w0_vals = np.linspace(w_hist_arr[0].min()-10000, w_hist_arr[0].max()+10000, 40)
w1_vals = np.linspace(w_hist_arr[1].min()-1000, w_hist_arr[1].max()+1000, 40)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
Loss_surface = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        w_temp = np.array([[W0[i,j]],[W1[i,j]]])
        preds = X_train_b.dot(w_temp)
        Loss_surface[i,j] = mse_loss(y_train, preds)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Loss_surface, alpha=0.6)
# plot trajectory
ax.plot(w_hist_arr[0], w_hist_arr[1], [mse_loss(y_train, X_train_b.dot(w)) for w in w_history], color='r', marker='o')
ax.set_xlabel('w0 (bias)')
ax.set_ylabel('w1 (slope)')
ax.set_zlabel('MSE Loss')
ax.set_title('Gradient Descent trajectory (parameter space)')
plt.tight_layout()
plt.savefig('gd_trajectory_3d.png')
plt.show()

# Save a small summary CSV of results
summary_rows = []
for k,v in results_exp3.items():
    lr, loss = k
    summary_rows.append({
        'lr': lr,
        'loss': loss,
        'test_mse': v['test_metrics']['mse'],
        'test_mae': v['test_metrics']['mae'],
        'test_r2': v['test_metrics']['r2']
    })
summary_df = pd.DataFrame(summary_rows).sort_values('test_mse')
summary_df.to_csv('results_summary.csv', index=False)
print('Saved results_summary.csv')

# Final notes: Print best overall
print('\nSummary of grid results (sorted by test_mse):')
print(summary_df)

# End of notebook/script
print('\nAll figures saved: exp1_lr_convergence.png, exp2_loss_comparison.png, model_fit_best_worst.png, gd_trajectory_3d.png')# Machine-learning-
