import numpy as np
import crocoddyl
import yaml
import os

import eagle_mpc
from eagle_mpc.utils.path import EAGLE_MPC_YAML_DIR
from eagle_mpc.utils.simulator import AerialSimulator

try:
    dataset = np.load("dataset.npy", allow_pickle=True)
except FileNotFoundError:
    dataset = []

for i in range(len(os.listdir("yaml"))):
    # Trajectory
    dt = 20  # ms
    useSquash = True
    robotName = 'hexacopter370_flying_arm_3'
    trajectoryName = 'eagle_catch_nc'
    mpcName = 'carrot'
    # mpcName = 'rail'

    yamlPath = "yaml/" + trajectoryName + '_' + str(i) + ".yaml"
    with open(yamlPath, "r") as file:
        data = yaml.safe_load(file)
    init, obj = [], []
    for stage in data['trajectory']['stages']:
        if stage['name'] == 'take_off':
            init = stage['costs'][0]['reference']
        if stage['name'] == 'pre_grasp':
            obj = stage['costs'][3]['position']

    trajectory = eagle_mpc.Trajectory()
    trajectory.autoSetup(yamlPath)
    problem = trajectory.createProblem(dt, useSquash, "IntegratedActionModelEuler")

    if useSquash:
        solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)
    else:
        solver = crocoddyl.SolverBoxFDDP(problem)

    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    solver.solve([], [], maxiter=100)

    mpcPath = EAGLE_MPC_YAML_DIR + "/" + robotName + "/mpc/mpc.yaml"
    if mpcName == 'rail':
        mpcController = eagle_mpc.RailMpc(solver.xs, dt, mpcPath)
    elif mpcName == 'weighted':
        mpcController = eagle_mpc.WeightedMpc(trajectory, dt, mpcPath)
    else:
        mpcController = eagle_mpc.CarrotMpc(trajectory, solver.xs, dt, mpcPath)

    mpcController.updateProblem(0)
    mpcController.solver.solve(solver.xs[:mpcController.problem.T + 1], solver.us[:mpcController.problem.T])
    mpcController.solver.convergence_init = 1e-3

    dtSimulator = 2
    simulator = AerialSimulator(mpcController.robot_model, mpcController.platform_params, dtSimulator, solver.xs[0])
    t = 0

    for i in range(0, int(problem.T * dt * 1.2)):
        mpcController.problem.x0 = simulator.states[-1]
        mpcController.updateProblem(int(t))
        mpcController.solver.solve(mpcController.solver.xs, mpcController.solver.us, mpcController.iters)
        control = np.copy(mpcController.solver.us_squash[0])
        simulator.simulateStep(control)
        t += dtSimulator

    # print(np.vstack(solver.xs).shape)
    # print(np.vstack(solver.us_squash).shape)
    # print(np.vstack(mpcController.solver.xs).shape)
    # print(np.vstack(mpcController.solver.us).shape)

    dataset.append({
        "initial_state": init,
        "target_pos": obj,
        "planned_path": np.vstack(solver.xs),
        "planned_control": np.vstack(solver.us_squash),
        "track_state": np.vstack(mpcController.solver.xs),
        "track_control": np.vstack(mpcController.solver.us)
    })

np.save("dataset.npy", dataset)
