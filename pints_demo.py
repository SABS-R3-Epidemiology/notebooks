import marimo

__generated_with = "0.10.19"
app = marimo.App(html_head_file="head.html")


@app.cell
def _():
    import time
    import marimo as mo
    import pints
    import pints.toy
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, pints, plt, time


@app.cell
def _(mo):
    mo.md(
        """
        ## Generating the data

        The data will be drawn from the Logistic Model.
        """
    )
    return


@app.cell
def _(mo):
    growth_rate = mo.ui.slider(0.1, 2.0, label='Select the true growth rate: ', step=0.1, value=0.5)
    carrying_capacity = mo.ui.slider(10, 50, label='Select the true carrying capacity: ')
    return carrying_capacity, growth_rate


@app.cell
def _(growth_rate):
    growth_rate
    return


@app.cell
def _(carrying_capacity):
    carrying_capacity
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Plot the data

        Below we plot the data generated according to the user specified parameter values. Noise has been added.
        """
    )
    return


@app.cell
def _(carrying_capacity, growth_rate, np, pints):
    times = np.arange(21)
    model = pints.toy.LogisticModel(initial_population_size=1)
    true_params = [growth_rate.value, carrying_capacity.value]
    data = model.simulate(true_params, times)
    data += np.random.normal(0, 0.5, data.shape)
    return data, model, times, true_params


@app.cell
def _(data, plt, times):
    fig = plt.figure(figsize=(3.5, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, data, 'o-', label='Data', color='k')
    ax.set_xlabel('Time')
    ax.set_ylabel('Output')
    ax.legend()

    fig
    return ax, fig


@app.cell
def _(mo):
    mo.md(
        """
        ## Run inference using MCMC and PINTS

        Choose the settings for the MCMC inference algorithm, and it will be run by Pints.

        Note that for simplicity the inference is being initialized at the true parameter values specified above.
        """
    )
    return


@app.cell
def _(mo):
    mcmc_algo = mo.ui.dropdown(['Haario Bardenet ACMC', 'Dream MCMC'], value='Haario Bardenet ACMC', label='Choose the MCMC method:')
    num_iterations = mo.ui.number(2, 1000, label='Choose the number of iterations (Maximum of 1000): ', value=250)
    button = mo.ui.run_button(label='Run MCMC')
    return button, mcmc_algo, num_iterations


@app.cell
def _(mcmc_algo):
    mcmc_algo
    return


@app.cell
def _(num_iterations):
    num_iterations
    return


@app.cell
def _(button):
    button
    return


@app.cell
def _(
    button,
    carrying_capacity,
    data,
    growth_rate,
    mcmc_algo,
    mo,
    model,
    num_iterations,
    pints,
    times,
):
    mo.stop(not button.value)

    problem = pints.SingleOutputProblem(model, times, data)
    likelihood = pints.GaussianLogLikelihood(problem)
    prior = pints.ComposedLogPrior(
        pints.GammaLogPrior(1, 0.1),
        pints.GammaLogPrior(1, 0.1),
        pints.GammaLogPrior(1, 0.1)
    )
    posterior = pints.LogPosterior(likelihood, prior)

    x0 = [growth_rate.value, carrying_capacity.value, 1]

    methods = {
        'Haario Bardenet ACMC': pints.HaarioBardenetACMC,
        'Dream MCMC': pints.DreamMCMC,
    }

    mcmc = pints.MCMCController(
        posterior,
        3,
        [x0, x0, x0],
        method=methods[mcmc_algo.value]
    )
    mcmc.set_max_iterations(num_iterations.value)

    chains = mcmc.run()
    return chains, likelihood, mcmc, methods, posterior, prior, problem, x0


@app.cell
def _(mo):
    mo.md(
        """
        ## Inference results

        These figures contain the pints plots showing the MCMC chains.
        """
    )
    return


@app.cell
def _(chains, plt):
    fig2 = plt.figure(figsize=(7, 3))

    num_params = 3

    label = ['Growth Rate', 'Carrying Capacity', r'$\sigma$']
    for i in range(num_params):
        ax2 = fig2.add_subplot(1, num_params, 1 + i)
        ax2.plot(chains[0, :, i], label='Chain 1')
        ax2.plot(chains[1, :, i], label='Chain 2')
        ax2.plot(chains[2, :, i], label='Chain 3')
        ax2.set_xlabel('MCMC iteration')
        if i == 0:
            ax2.legend()

        ax2.set_ylabel(label[i])

    fig2.set_tight_layout(True)

    fig2
    return ax2, fig2, i, label, num_params


@app.cell
def _(chains, data, model, plt, times):
    fig3 = plt.figure(figsize=(3.5, 2))

    ax3 = fig3.add_subplot(1, 1, 1)

    for j, param in enumerate(chains[0, :, :]):
        ax3.plot(times, model.simulate(param[:2], times), alpha=0.5, color='tab:blue', label='Model Fits' if j == 0 else None)

    ax3.plot(times, data, 'o-', color='k', label='Data')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Output')
    ax3.legend()

    fig3
    return ax3, fig3, j, param


@app.cell
def _(mo):
    mo.md(r"""## Speed test of slower inference (HES1 Michaelis-Menten Model)""")
    return


@app.cell
def _(mo):
    button_slow = mo.ui.run_button(label='Run slower MCMC')
    return (button_slow,)


@app.cell
def _(button_slow):
    button_slow
    return


@app.cell
def _(button_slow, mo, np, pints, time):
    mo.stop(not button_slow.value)


    times_slow = np.arange(101)
    model_slow = pints.toy.Hes1Model()
    true_params_lv = [1, 1, 1, 1,]
    data_slow = model_slow.simulate(true_params_lv, times_slow)
    data_slow += np.random.normal(0, 0.5, data_slow.shape)

    t0 = time.time()

    problem_slow = pints.SingleOutputProblem(model_slow, times_slow, data_slow)
    likelihood_slow = pints.GaussianLogLikelihood(problem_slow)
    prior_slow = pints.ComposedLogPrior(
        pints.GammaLogPrior(1, 0.1),
        pints.GammaLogPrior(1, 0.1),
        pints.GammaLogPrior(1, 0.1),
        pints.GammaLogPrior(1, 0.1),
        pints.GammaLogPrior(1, 0.1),
    )
    posterior_slow = pints.LogPosterior(likelihood_slow, prior_slow)

    mcmc_slow = pints.MCMCController(
        posterior_slow,
        1,
        [[1, 1, 1, 1, 1,]],
        method=pints.HamiltonianMCMC
    )
    mcmc_slow.set_max_iterations(30)

    _ = mcmc_slow.run()

    t1 = time.time()
    return (
        data_slow,
        likelihood_slow,
        mcmc_slow,
        model_slow,
        posterior_slow,
        prior_slow,
        problem_slow,
        t0,
        t1,
        times_slow,
        true_params_lv,
    )


@app.cell
def _(t0, t1):
    '{} Seconds'.format(t1 - t0)
    return


if __name__ == "__main__":
    app.run()
