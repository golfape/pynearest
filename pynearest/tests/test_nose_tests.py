from nose.tools import with_setup

def test_not_much():
    """ Just making sure nosetests is working """
    pass

def setup_func():
    "set up test fixtures"

def teardown_func():
    "tear down test fixtures"

@with_setup(setup_func,teardown_func)
def test_with_setup():
    pass

