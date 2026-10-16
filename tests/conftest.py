import pytest


@pytest.fixture
def sample_markdown() -> str:
    return """\
# Chapter 1: Software Engineering

Software engineering is the systematic application of engineering approaches
to the development of software. It encompasses requirements, design,
construction, testing and maintenance.

## 1.1 Requirements

Requirements engineering focuses on eliciting, documenting and validating
what stakeholders need. Functional requirements describe behaviour;
non-functional requirements describe qualities like performance.

## 1.2 Design

Design translates requirements into a blueprint. Architectural design sets
the high-level structure; detailed design specifies modules and interfaces.

# Chapter 2: Agile Methods

Agile methods embrace iterative delivery and continuous feedback. Scrum
organises work into sprints; XP emphasises engineering practices such as
pair programming and test-driven development.
"""
