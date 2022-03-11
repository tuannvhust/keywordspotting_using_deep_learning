import util
import scenario

def main():
    args = util.get_args()
    method = getattr(scenario, args.scenario)
    method(args)

if __name__ == '__main__':
    main()