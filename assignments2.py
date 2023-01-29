import evo
import pandas as pd
import random as rnd
import numpy as np
from itertools import repeat
import os
import random
import time

# read tas and sections dfs
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "."))
tas = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'tas.csv'))
sections = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'sections.csv'))


def overallocation(sol):
    """ get overallocation score

    :param sol: (array) array of assignments
    :return over (int): overallocation score
    """

    # get totals number of student assignments
    total = list(sol.sum(axis=1))

    # make list of each student's max_assigned interest
    m_assigned = list(tas["max_assigned"])

    # map calculation to get overassigned count for each student and sum all overassignments
    counter = list(map(lambda x: int(x[0]) - int(x[1]) if int(x[0]) > int(x[1]) else 0, list(zip(total, m_assigned))))
    over = sum(counter)

    # return overallocation count
    return over

def undersupport(sol):
    """ get undersupport score

    :param sol: (array) array of assignments
    :return over (int): undersupport score
    """

    # get totals number of assignments for each lab
    total = list(sol.sum(axis=0))

    # make list of each section's preference for minimum tas
    min_ta = list(sections["min_ta"])

    # calculate number of tas undersupporting specific labs and sum all of these penalty points
    counter = list(map(lambda x: (x[1] - x[0]) if (x[0] < x[1]) else 0, list(zip(total, min_ta))))
    under = sum(counter)

    # return understaffed score
    return under

def conflicts(sol):
    """ get conflicts score

    :param sol: (array) array of assignments
    :return over (int): conflicts score
    """
    # cant be done functionally

    # make list of sections day/time
    secs = list(sections["daytime"])

    # count number of times a ta is assigned to two labs at the same time
    con = 0
    for i in range(len(sol)):
        times = {}
        for j in range(len(sol[i])):
            if sol[i][j] == 1:
                t = secs[j]
                if t in times.keys():
                    con += 1
                    break
                else:
                    times[t] = 1

    # return score
    return con


def fun(zipper_line, searchval):
    """ return count of times someone is assigned to a lab that they don't prefer or aren't able

    :param zipper_line: (list of 2) list containing preferences list and lab assignment list for 1 student
    :param searchval: (int) value for np to search for
    :return: (int) count of times someone is assigned to a lab that they don't prefer or aren't able
    """

    # make np array of a students' assignments and preferences
    assigns = np.array(zipper_line[0])
    prefs = np.array(zipper_line[1])

    # get indexes where a student is assigned to a lab or where their preference is
    ii = np.where(assigns == searchval)[0]
    iii = np.where(prefs == searchval)[0]

    # count number of times indexes of preferences are in actual assignments
    return sum(np.in1d(iii, ii))

def unpreferred(sol):
    """ get unpreferred score

    :param sol: (array) array of assignments
    :return over (int): unpreferred score
    """

    # get ta preferences values as a list of lists
    # make all "willing" spots equal 1 so we can search for unpreferred
    df = (tas.copy().drop(columns=["ta_id", "name", "max_assigned"])).values
    pref = [[1 if x == "W" else 0 for x in lst] for lst in df]

    # zip together 2 lists of lists (preferences and actual assignments)
    zipper = list(zip(sol, pref))

    # map function onto each student, sum number of unpreferrences
    unpref = sum(list(map(fun, zipper, repeat(1))))

    # return score
    return unpref

def unwilling(sol):
    """ get unwilling score

    :param sol: (array) array of assignments
    :return over (int): unwilling score
    """

    # get ta preferences values as a list of lists
    # make all "unwilling" spots equal 1 so we can search for unwilling
    df = (tas.copy().drop(columns=["ta_id", "name", "max_assigned"])).values
    df = [[1 if x == "U" else 0 for x in lst] for lst in df]

    # zip together 2 lists of lists (preferences and actual assignments)
    zipper = list(zip(sol, df))

    # map function onto each student, sum number of unwillings
    unwill = sum(list(map(fun, zipper, repeat(1))))

    # return score
    return unwill


def swap_rows(solutions):

    solution = solutions[0]

    # randmonly shuffle rows in array
    np.random.shuffle(solution)

    return solution


def shuffler(solutions):

    # shuffles the items within the solutions for a random list
    solution = solutions[0]

    # picks a random ta
    i = rnd.randrange(0, len(solution))

    # shuffles lab assignments for singular ta
    np.random.shuffle(solution[i])
    return solution

def conflict_helper(solutions):

    # if a ta is assigned to all labs in one time slot, randomly turn remove one of the assingments
    if solutions[1] == 1 and solutions[2] == 1 and solutions[3] == 1:
        solutions[random.randrange(1, 4)] = 0
    if solutions[4] == 1 and solutions[5] == 1:
        solutions[random.randrange(1, 3) + 3] = 0
    if solutions[6] == 1 and solutions[7] == 1 and solutions[8] == 1:
        solutions[random.randrange(1, 4) + 5] = 0
    if solutions[10] == 1 and solutions[11] == 1:
        solutions[random.randrange(1, 3) + 9] = 0
    if solutions[12] == 1 and solutions[13] == 1 and solutions[14] == 1:
        solutions[random.randrange(1, 4) + 11] = 0
    if solutions[15] == 1 and solutions[16] == 1:
        solutions[random.randrange(1, 3) + 14] = 0
    return list(solutions)

def remove_conflicts(solutions):

    # randomly removes one assignment from ta lab time if they are assigned to all slots in that time
    return np.array(list(map(conflict_helper, solutions[0])))

def fix_unwilling(solutions):

    # create array of 1s and 0s lab assignments
    df = (tas.copy().drop(columns=["ta_id", "name", "max_assigned"])).values
    sol = solutions[0]

    # iterate for each assignment
    for i in range(len(sol)):
        for idx in range(len(sol[i])):

            # use random to do something 90% of the time
            val = random.randrange(1, 11)
            if val != 10:

                # 90% of the time set a TA's unwilling assignment to 0
                if df[i][idx] == "U":
                    sol[i][idx] = 0

    return sol

def fix_unpreferred(solutions):
    # create array of 1s and 0s lab assignments
    df = (tas.copy().drop(columns=["ta_id", "name", "max_assigned"])).values
    sol = solutions[0]

    # iterate for each assignment
    for i in range(len(sol)):
        for idx in range(len(sol[i])):

            # use random to do something 50% of the time
            val = random.randrange(1, 11)
            if val < 6:

                # 50% of the time set a TA's willing assignment to 0
                if df[i][idx] == "W":
                    sol[i][idx] = 0

    return sol

def undersupportagent(solutions):

    # get total number of tas for each section
    solution = solutions[0]
    section_sums = [sum(x) for x in zip(*solution)]

    # iterate through each section
    for each in section_sums:

        # check if section is assigned less than 3 TAs
        if each < 3:
            i = 0

            # while section has less than 3 assigned TAs
            while i < 3:

                # get random ta
                randTA = solution[section_sums.index(each)][i]

                # if ta is already assigned, pass
                if randTA == 1:
                    pass

                # assign first available TAs to section
                elif randTA == 0:
                    solution[section_sums.index(each)][i] = 1
                i += 1

    return solution

def weave_sol(solutions):

    # weaves rows of two solutions randomly
    sol1, sol2 = solutions[0], solutions[1]
    new_sol = np.array([list(sol1[i]) if i % 2 == 1 else list(sol2[i]) for i in range(len(sol1))])

    return new_sol

def switch_ones(solutions):

    solution = solutions[0]

    # for each TA
    for row in solution:

        # assign TA to two random sections
        indices = rnd.sample(range(0, 17), 2)
        row[indices[0]] = 0
        row[indices[1]] = 0

    return solution

def undersupport_fix(solutions):
    solution = solutions[0]

    # get total number of TAs for each lab
    sums = list(np.sum(solution, axis=0))

    # for each lab
    for idx in range(len(sums)):

        # if a lab has no TAs assigned
        if sums[idx] == 0:

            # get 3 random TAs and assign them to this lab
            indices = rnd.sample(range(0, 42), 4)
            solution[indices[0]][idx] = 1
            solution[indices[1]][idx] = 1
            solution[indices[2]][idx] = 1
            solution[indices[3]][idx] = 1

    return solution

def main():

    # create population
    E = evo.Environment()

    # register the fitness criteria (objects)
    E.add_fitness_criteria("overallocation", overallocation)
    E.add_fitness_criteria("undersupport", undersupport)
    E.add_fitness_criteria("conflicts", conflicts)
    E.add_fitness_criteria("unwilling", unwilling)
    E.add_fitness_criteria("unpreferred", unpreferred)

    # register all agents
    E.add_agent("swap_rows", swap_rows, 1)
    E.add_agent("shuffler", shuffler, 1)
    E.add_agent("remove_conflicts", remove_conflicts, 1)
    E.add_agent("fix_unwilling", remove_conflicts, 1)
    E.add_agent("fix_unpreferred", remove_conflicts, 1)
    E.add_agent("undersupportagent", undersupportagent, 1)
    E.add_agent("weave_sol", weave_sol, 2)
    E.add_agent("switch_ones", switch_ones, 1)
    E.add_agent("undersupport_fix", undersupport_fix, 1)


    # seed the population with an initial solution
    rows, cols = (len(tas), len(sections))
    first_sol = np.array([[rnd.randrange(0, 2) for i in range(cols)] for j in range(rows)])
    E.add_solution(first_sol)

    # run the evolver
    E.evolve(n = 1000000)

if __name__ == "__main__":
    main()
