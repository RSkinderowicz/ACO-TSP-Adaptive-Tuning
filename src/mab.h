#pragma once

/* Few algorithms for solving Multi-Armed Bandit problem */

#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>
#include <vector>

#include "rand.h"

namespace MultiArmedBandit {

class Policy { 

protected:
    std::size_t arms_;
    std::vector<int> arm_pulls_;
    std::vector<double> arm_rewards_;
    int total_pulls_ = 0;

    virtual std::size_t select_arm() = 0;

    virtual void reward_arm(std::size_t arm, double reward) = 0;

    static size_t random_choice(const std::vector<double> &probs) {
        auto r = get_rng().next_float();
        auto n = probs.size();
        auto selected = n - 1;
        for (size_t choice = 0; choice < n; ++choice) {
            if (r < probs[choice]) {
                return choice;
            }
            r -= probs[choice];
        }
        return selected;
    }

public:

    Policy(std::size_t num_arms)
        : arms_(num_arms),
          arm_pulls_(num_arms, 0),
          arm_rewards_(num_arms, 0) {}

    virtual ~Policy() = default; // Ensure virtual destructor

    std::size_t pull() {
        size_t arm;
        #pragma omp critical
        {
            arm = select_arm();
            ++arm_pulls_.at(arm);
            ++total_pulls_;
        }
        return arm;
    }    

    void update(std::size_t arm, double reward) {
        #pragma omp critical
        {
            arm_rewards_.at(arm) += reward;
            reward_arm(arm, reward);
        }
    }

    virtual void print_summary() {
        std::cout << "# arms: " << arms_ << "\n";
        std::cout << "arm\ttimes pulled\ttotal reward\n";
        for (size_t arm = 0; arm < arms_; ++arm) {
            std::cout << arm << "\t" << arm_pulls_[arm] << "\t" << arm_rewards_[arm] << "\n";
        }
    }

    const std::vector<int>& get_arm_pulls() const { return arm_pulls_; }

    const std::vector<double>& get_arm_rewards() const { return arm_rewards_; }
};


class Uniform : public Policy {
protected:
    std::size_t select_arm() override { return get_rng().next_uint32(arms_); }
    void reward_arm(std::size_t /*arm*/, double /*reward*/) override {}

public:
    Uniform(std::size_t num_arms) : Policy(num_arms) {}
};

/* epsilon-greedy */
class EpsilonGreedy : public Policy {
    double epsilon_;

    std::vector<double> est_arm_rewards_;

protected:

    std::size_t select_arm() override {
        if (get_rng().next_float() > epsilon_) {  // Greedy choice -- exploitation
            auto max_el_it = std::max_element(est_arm_rewards_.begin(), est_arm_rewards_.end());
            return std::distance(est_arm_rewards_.begin(), max_el_it);
        }
        // Exploration
        return get_rng().next_uint32(arms_);
    }

    void reward_arm(std::size_t chosen_arm, double reward) override {
        double n = static_cast<double>(arm_pulls_.at(chosen_arm));
        double value = est_arm_rewards_[chosen_arm];
        est_arm_rewards_[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward;
    }

public:
    EpsilonGreedy(std::size_t n_arms, double epsilon = 0.05) : Policy(n_arms),
                 epsilon_(epsilon),
                 est_arm_rewards_(n_arms, 0.0)
    {}

    void print_summary() override {
        Policy::print_summary();

        std::cout << "\nArms' rewards: ";
        for (auto value : est_arm_rewards_) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
};



/* Follow the perturbed leader algorithm */
class FPL : public Policy {
    double eta_ = 0.5;

protected:

    std::size_t select_arm() override {
        double max_value = std::numeric_limits<double>::min();
        size_t selected = 0;
        for (size_t arm = 0; arm < arms_; ++arm) {
            auto value = arm_rewards_[arm] + get_rng().next_exp(eta_);
            if (value > max_value) {
                max_value = value;
                selected = arm;
            }
        }
        return selected;
    }

    void reward_arm(std::size_t /*arm*/, double /*reward*/) override {}

public:
    FPL(size_t arms, double eta = 0.5)
        : Policy(arms),
          eta_(eta)
    {}
};


class Exp3 : public Policy {

    std::vector<double> weights_;
    std::vector<double> probs_;
    double gamma_;

protected:

    std::size_t select_arm() override {
        return random_choice(probs_);
    }

    void reward_arm(std::size_t arm, double reward) override {
        double est_reward = reward / probs_.at(arm);
        weights_[arm] *= exp(gamma_ * est_reward / arms_);

        const auto total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        for (size_t i = 0; i < arms_; ++i) {
            probs_[i] = (1 - gamma_) * weights_[i] / total + gamma_/arms_;
        }
    }

public:
    Exp3(size_t arms = 4)
        : Policy(arms),
          weights_(arms, 1),
          probs_(arms, 1. / arms),
          gamma_(0.1)
    {
    }

    void print() {
        std::cout << "Exp3:\n"
        << "\tArms pulls: ";
        for (auto x : arm_pulls_) {
            std::cout << x << " ";
        }
        std::cout << "\tWeights: ";
        for (auto x : weights_) {
            std::cout << x << " ";
        }
    }
};


class Exp3P : public Policy {
    std::vector<double> weights_;
    std::vector<double> probs_;
    size_t T_;
    double alpha_;
    double gamma_;

protected:

    std::size_t select_arm() override {
        return random_choice(probs_);
    }

    void reward_arm(std::size_t arm, double reward) override {
        double est_reward = reward / probs_.at(arm);
        weights_[arm] *= exp(gamma_ / (3.*arms_) * (est_reward + alpha_ / (probs_.at(arm) * sqrt(arms_ * T_))));

        const auto total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        for (size_t i = 0; i < arms_; ++i) {
            probs_[i] = (1 - gamma_) * weights_[i] / total + gamma_/arms_;
        }
    }

public:

    /* T is the number of times the arms will be pulled (selected), i.e. length
    of the time horizon. */
    Exp3P(size_t arms, size_t T)
        : Policy(arms),
          weights_(arms),
          probs_(arms, 1. / arms),
          T_(T)
    {
        double delta = 0.5;
        alpha_ = 2 * sqrt(log(arms * T / delta));
        gamma_ = std::min(3./5, 2 * sqrt(3./5 * arms * log(arms)/T));
        double w1 = exp(alpha_ * gamma_ / 3 * sqrt(T / (1.*arms)));
        for (auto &w : weights_) {
            w = w1;
        }
        std::cout << "alpha_ = " << alpha_ << " gamma: " << gamma_ << " w1: " << w1 << std::endl; 
    }

    void print_summary() override {
        std::cout << "\nWeights: ";
        for (auto x : weights_) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
};


/**
 * Reinforcement Comparison algorithm
 * As described in:
 * Kuleshov, Volodymyr, and Doina Precup. "Algorithms for multi-armed bandit
 * problems." arXiv preprint arXiv:1402.6028 (2014).
*/
class RC : public Policy {
    std::vector<double> preferences_;
    std::vector<double> probs_;
    double alpha_;
    double beta_;
    double mean_reward_ = 0;

protected:

    std::size_t select_arm() override {
        return random_choice(probs_);
    }

    void reward_arm(std::size_t arm, double reward) override {
        preferences_[arm] += beta_ * (reward - mean_reward_);
        mean_reward_ = (1 - alpha_) * mean_reward_ + alpha_ * reward;

        double total = 0;
        for (size_t i = 0; i < arms_; ++i) {
            probs_[i] = exp(preferences_[i]);
            total += probs_[i];
        }
        for (size_t i = 0; i < arms_; ++i) {
            probs_[i] /= total;
        }
    }

public:
    RC(std::size_t num_arms, double alpha = 0.01, double beta = 0.9)
        : Policy(num_arms),
          preferences_(num_arms, 0),
          probs_(num_arms, 0),
          alpha_(alpha),
          beta_(beta)
    {
    }


    void print_summary() override {
        Policy::print_summary();

        std::cout << "Mean reward: " << mean_reward_ << "\n";
        std::cout << "Preferences: ";
        for (auto x : preferences_) {
            std::cout << x << " ";
        }
        std::cout << "\nProbabilities: ";
        for (auto x : probs_) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
};


/**
 * Upper Confidence Bounds (UCB) algorithm for stochastic MAB
*/
class UCB1 : public Policy {
    std::vector<double> est_arm_rewards_;

protected:

    std::size_t select_arm() override {
        for (std::size_t arm = 0; arm < arms_; ++arm) {
            if (arm_pulls_[arm] == 0) {
                return arm;
            }
        }

        double max_ucb = std::numeric_limits<double>::min();
        size_t selected = 0;

        for (std::size_t arm = 0; arm < arms_; ++arm) {
            auto bonus = std::sqrt((2 * std::log(total_pulls_)) / arm_pulls_[arm]);
            auto ucb = est_arm_rewards_[arm] + bonus;

            if (ucb > max_ucb) {
                max_ucb = ucb;
                selected = arm;
            }
        }
        return selected;
    }

    void reward_arm(std::size_t chosen_arm, double reward) override {
        double n = static_cast<double>(arm_pulls_.at(chosen_arm));
        double value = est_arm_rewards_[chosen_arm];
        est_arm_rewards_[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward;
    }

public:
    UCB1(std::size_t n_arms) : Policy(n_arms),
                               est_arm_rewards_(n_arms, 0.0) {}

    void print_summary() override {
        Policy::print_summary();

        std::cout << "\nArms' rewards: ";
        for (auto value : est_arm_rewards_) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
};


/**
 * Tuned version of the UCB1 alg. as proposed in:
 * 
 * Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time analysis of
 * the multiarmed bandit problem." Machine learning 47 (2002): 235-256.
*/
class UCB1Tuned : public Policy {
    std::vector<double> est_arm_rewards_;
    std::vector<double> squared_rewards_;

protected:

    std::size_t select_arm() override {
        using namespace std;

        for (size_t arm = 0; arm < arms_; ++arm) {
            if (arm_pulls_[arm] == 0) {
                return arm;
            }
        }

        double max_ucb = numeric_limits<double>::min();
        size_t selected = 0;

        for (size_t arm = 0; arm < arms_; ++arm) {
            auto arm_pulls = arm_pulls_[arm];
            auto avg_reward = est_arm_rewards_[arm];

            auto variance = squared_rewards_[arm] / arm_pulls 
                          - avg_reward * avg_reward 
                          + sqrt(2 * log(total_pulls_) / arm_pulls);

            auto bonus = sqrt(std::log(total_pulls_) / arm_pulls) * std::min(0.25, variance);

            auto ucb = avg_reward + bonus;

            if (ucb > max_ucb) {
                max_ucb = ucb;
                selected = arm;
            }
        }
        return selected;
    }

    void reward_arm(std::size_t chosen_arm, double reward) override {
        double n = static_cast<double>(arm_pulls_.at(chosen_arm));
        double value = est_arm_rewards_[chosen_arm];
        est_arm_rewards_[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward;
        squared_rewards_[chosen_arm] += reward*reward;
    }

public:
    UCB1Tuned(std::size_t n_arms)
        : Policy(n_arms),
          est_arm_rewards_(n_arms, 0.0),
          squared_rewards_(n_arms, 0.0)
    {
    }

    void print_summary() override {
        Policy::print_summary();

        std::cout << "\nArms' rewards: ";
        for (auto value : est_arm_rewards_) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
};

};