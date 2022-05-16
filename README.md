# Competition_Olympics-Football

## Environment

<img src=https://jidi-images.oss-cn-beijing.aliyuncs.com/jidi/env73.gif width=600>


Check details in Jidi Competition [第六届北京科技大学RoboCup校内赛之足球机器人的策略强化学习比赛](http://www.jidiai.cn/compete_detail?compete=19)


### Olympics-Integrated
<b>Tags: </b>Partial Observation; Continuous Action Space; Continuous Observation Space

<b>Introduction: </b>Agents participate in the Olympic Games. In this series of competitions, two agents participate in 1vs1 football game.

<b>Environment Rules:</b> 
1. This game has two sides and both sides control an elastic ball agent with the same mass and radius.
2. Agents can collide with each other or walls, but they might lose a certain speed. The agent has its own energy, and the energy consumed in each step is directly proportional to the applied driving force and displacement.
3. The energy of the agent recovers at a fixed rate at the same time. If the energy decays to zero, the agent will be tired, resulting in failure to apply force.
4. When one side scores a goal or the environment reaches the maximum number of 500 steps, the environment ends and the side with the advanced ball wins.


<b>Action Space: </b>Continuous, a matrix with shape 2*1, representing applied force and steering angle respectively.

<b>Observation: </b>A dictionary with keys 'obs' and 'controlled_player_index'. The value of 'obs' contains a 2D matrix with shape of 40x40 and other game-releated infomation. The 2D matrix records the view of agent along his current direction. Agent can see walls, marking lines, opponents and other game object within the vision area. The value of 'controlled_player_index' is the player id of the game. The side information includes energy left and a game-switching flags.

<b>Reward: </b>Each team obtains a +100 reward when scoring the ball into the opponent's goal, otherwise 0 point.

<b>Environment ends condition: </b>The game ends when reaching maximum number of 500 steps or one side scores a goal.

<b>Registration: </b>Go to (http://www.jidiai.cn/compete_detail?compete=19).


---
## Dependency

>conda create -n olympics python=3.8.5

>conda activate olympics

>pip install -r requirements.txt

---

## Run a game

>python olympics_engine/main.py --map football

## How to test submission

You can locally test your submission. At Jidi platform, we evaluate your submission as same as *run_log.py*

For example,

>python run_log.py --my_ai "random" --opponent "random"

in which you are controlling agent 1 which is green.

---

## Ready to submit

Random policy --> *agents/random/submission.py*
