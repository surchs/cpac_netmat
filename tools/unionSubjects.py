'''
Created on Mar 9, 2013

@author: surchs

script to combine two subject lists into one list
lists may only contain subject names and only one subject name per line
'''
import sys


def Main(subList1, subList2, subListCombined):
    # Read the inputs
    sL1 = open(subList1, 'rb')
    l1subjects = sL1.readlines()
    sL2 = open(subList2, 'rb')
    l2subjects = sL2.readlines()

    # Prepare the output string
    combinedSubs = ''
    combinedSubCount = 0

    # Merge the first to the second
    for sub1 in l1subjects:
        subject1 = sub1.strip()
        # Check if in list
        if not subject1 in l2subjects:
            # unique, add it
            # print('List1 subject ' + subject1 + ' is unique and gets added')
            combinedSubs = (combinedSubs + subject1 + '\n')

            combinedSubCount += 1
        else:
            print('List1 subject ' + subject1 + ' is already in list 2')

    # Now merge the second list back
    for sub2 in l2subjects:
        subject2 = sub2.strip()
        # Sanity check:
        if subject2 in combinedSubs:
            # uhoh
            print('List2 subject ' + subject2
                  + ' shouldn\'t be in this list...')
            continue
        else:
            print('List2 subject ' + subject2 + ' can enter the list!')
            combinedSubs = (combinedSubs + subject2 + '\n')

            combinedSubCount += 1

    # Now print that out
    outList = open(subListCombined, 'wb')
    outList.writelines(combinedSubs)
    outList.close()

    print('Saved a list of ' + str(combinedSubCount) + ' subjects')
    print('Done')


if __name__ == '__main__':
    subList1 = sys.argv[1]
    subList2 = sys.argv[2]
    subListCombined = sys.argv[3]
    Main(subList1, subList2, subListCombined)
    pass
