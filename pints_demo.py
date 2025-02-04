import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pints
    import pints.toy
    import pints.plot
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, pints, plt


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
    growth_rate = mo.ui.slider(0.1, 1.0, label='Select the true growth rate: ', step=0.1)
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
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, data, 'o-', label='Data')
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
    mcmc_algo = mo.ui.dropdown(['Haario Bardenet ACMC', 'Population MCMC'], value='Haario Bardenet ACMC', label='Choose the MCMC method:')
    num_iterations = mo.ui.number(2, 1000, label='Choose the number of iterations (Maximum of 1000): ', value=250)
    return mcmc_algo, num_iterations


@app.cell
def _(mcmc_algo):
    mcmc_algo
    return


@app.cell
def _(num_iterations):
    num_iterations
    return


@app.cell
def _(
    carrying_capacity,
    data,
    growth_rate,
    mcmc_algo,
    model,
    num_iterations,
    pints,
    times,
):
    problem = pints.SingleOutputProblem(model, times, data)
    likelihood = pints.GaussianLogLikelihood(problem)
    prior = pints.UniformLogPrior([0, 0, 0], [100, 100, 100])
    posterior = pints.LogPosterior(likelihood, prior)

    x0 = [growth_rate.value, carrying_capacity.value, 1]

    methods = {
        'Haario Bardenet ACMC': pints.HaarioBardenetACMC,
        'Population MCMC': pints.PopulationMCMC,
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
def _(chains, pints):
    fig2, ax2 = pints.plot.trace(chains)
    fig2
    return ax2, fig2


@app.cell
def _(chains, num_iterations, pints, problem):
    fig3, ax3 = pints.plot.series(chains[0, num_iterations.value//2:, :], problem)
    fig3
    return ax3, fig3


if __name__ == "__main__":
    app.run()
