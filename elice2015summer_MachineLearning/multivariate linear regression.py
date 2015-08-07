import statsmodels.api
import numpy

def main():
    (N, X, Y) = read_data()

    results = do_multivariate_regression(N, X, Y)
  
    effective_variables = get_effective_variables(results)
    print(effective_variables)


def read_data():
    # 1
    N = 0
    X =  []
    Y = []
    with open("students.dat") as f:
        next(f)
        for line in f:
            N = N + 1
            temp = [float(x) for x in line.strip().split(" ") ] 
            X.append([temp[0], temp[1], temp[2], temp[3], temp[4]])
            Y.append([temp[5]])
    # X must be numpy.array in (30 * 5) shape.
    # Y must be 1-dimensional numpy.array.
    X = numpy.array(X)
    Y = numpy.array(Y)
    return (N, X, Y)

def do_multivariate_regression(N, X, Y):
    # 2
    X = statsmodels.api.add_constant(X)
    results = statsmodels.api.OLS(Y,X).fit()
    return results


def get_effective_variables(results):
    eff_vars = []
    vars  = [ x for x in results.pvalues]
    for i in range(len(vars)):
        if vars[i] < 0.05:
            eff_vars.append("x" + str(i+1))
    return eff_vars

def print_students_data():
    with open("students.dat") as f:
        for line in f:
            print(line)

if __name__ == "__main__":
    main()