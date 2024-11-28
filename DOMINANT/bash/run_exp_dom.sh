#!/bin/bash

echo "Running training.sh"
./train.sh

cd ../bash
echo "Running find_scores.sh"
./find_scores.sh

cd ../bash
echo "Running test.sh"
./test.sh

echo "End experiment"
cd ../bash
