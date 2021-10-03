.. sknet documentation master file, created by
   sphinx-quickstart on Fri Mar  5 05:44:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Development
===========

This is a guide for anyone interested in helping the development of the sknet. The library is an
open-source project and therefore depends on the community to keep existing, everyone is welcome
to help us improve.

How to contribute?
------------------

There are several ways of contributing for the sknet. Below we state those ways in ascending order
of complexity.

Opening an issue
^^^^^^^^^^^^^^^^
We oficially use the Issue Tracker on our github repo to hold up every new feature request and bug
tracking. Therefore, you can open up an issue to:

- Warn us about some bug on the library
- Warn us about documentation errors
- Request a new feature or change for one already implemented algorithm
- Request a brand new algorithm

We provide a basic template for issues on the ``templates`` folder inside our Github repo, please
refer to it before opening the issue so you can provide us with all of the information we need to
evaluate and (possibly) work on your issue.

Contributing to the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you found some problem on the documentation, such as wrong information, typos or something that
is missing or could be better explained, please, open an issue about it. After that, if you want to
correct the documentation yourself, feel free to open up a Pull Request with your correction. Please
remember to cite your issue on the Pull Request.

Solving an issue
^^^^^^^^^^^^^^^^
If you find an issue on the repo and thinks you can solve it, we encourage you to do so. Just verify
previously if no one is already assigned that issue. If this is not the case, then you can ask, on the
issue comments, to be assigned it.

Once you finished your implementation, open up a Pull Request on the repo. Your Pull Request will be
reviewed. Be aware that several iterations of revision may be required before the code is merged. We
encourage you to see the revision as a chat between two (or more) people trying to deliver the best
possible product to the people using the library.

Pull Requests
-------------
For those interested into opening Pull Requests for new features on the code, we will briefly describe
some of the things you must pay attention to.

First of all, one template for Pull Requests is available on the ``templates`` folder inside the Github
repo. Regarding to your code, some restrictions must be satisfied:

- Every Pull Request branch must be made from and merged to the ``develop`` branch
- Every new class or method must be unittested with pytest. We will not accept additions to the repo that reduces our coverage without a good reason for doing so
- Every public method must have a docstring using the numpy docstring style
- Every code must adhere to the PEP8. We suggest using flake8 to assess your style
- Performance improvements must contain benchmarks results
- We value good-sense when documenting methods
- Every change to modules with interface to users must have an entry on the documentation

Our Continous Integration pipeline will help you ensure most of those aspects. However, we strongly
encourage you to run those tests on your machine before submitting the Pull Request to avoid overhead
on the CI.

Doubts?
-------

If any doubt remain, please fell free to contact any of the developers.
