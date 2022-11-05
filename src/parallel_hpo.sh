#!/usr/bin/env bash
# Shell script to run main.py (HPO) in parallel processes
# Uses default parameters of HPO

# Arguments:
    # 1  Configuration file
    # 2  Maximal amount of processes
    #    If not set, there will start one process for every available CPU
    #    Attention: One process can still start multiple threads!

# Get number of CPUs
cpus=$(grep -c ^processor /proc/cpuinfo)

# Init number of processes
processes=$cpus

#----------------------
# Check arguments
#----------------------

if [ $# -gt 2 ]
  then
    echo "Too many arguments. Only arguments #1 and #2 will be used."
    echo
fi

# Check first argument
if ! [ -z $1 ]
then
    # Check if first argument is a string
    re='^[a-z._0-9-]+$'
    if ! [[ $1 =~ $re ]] ; then
        echo "Error argument #1: not a valid string"
        exit 1
    fi
    arg_1="-c $1"
fi

# Check second argument
if ! [ -z $2 ]
then
    # Check if second argument is a number
    re='^[-]*[0-9]+$'
    if ! [[ $2 =~ $re ]] ; then
        echo "Error argument #2: not a number"
        exit 1
    fi
    # Check if first argument is greater than zero
    if [ $2 -lt 1 ]
    then
        echo "Error argument #2: value less than 1"
        exit 1
    fi

    # Determine new number of processes
    if [ $2 -gt $cpus ]
    then
        echo "Only $cpus CPUs are available. (Not $2)"
    fi

    if [ $2 -lt $cpus ]
    then
        processes=$2
    fi
fi

#----------------------
# End check arguments
#----------------------

# Start processes
echo "----------------------------------------------"
echo "Starting $processes parallel HPO processes in the background..."
echo "----------------------------------------------"
for (( p=1; p<=$processes; p++ ))
do
    # Run python script in background
    echo "Process $p/$processes started."
    python3 main.py $arg_1 &

    # Wait for user to see what is going on
    # Also necessary for the creation of new mlflow experiments
    # Fails if done parallely without database backend
    sleep 3
done
