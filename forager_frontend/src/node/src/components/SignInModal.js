import React from "react";
import {
  Button,
  Form,
  FormGroup,
  Input,
  Modal,
  ModalBody
} from "reactstrap";

const SignInModal = ({
  isOpen,
  toggle,
  loginUsername,
  loginPassword,
  setLoginUsername,
  setLoginPassword,
  login
}) => {
  return (
    <Modal
      isOpen={isOpen}
      toggle={toggle}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
    >
      <ModalBody>
        <div className="m-xl-4 m-3">
          <div className="text-center mb-4">
            <h4 className="h3 mb-1">Welcome back</h4>
            <span>Enter your account details below</span>
          </div>
          <Form>
            <FormGroup>
              <Input
                type="email"
                placeholder="Email Address"
                value={loginUsername}
                onChange={(e) => setLoginUsername(e.target.value)}
              />
            </FormGroup>
            <FormGroup>
              <Input
                type="password"
                placeholder="Password"
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
              />
            </FormGroup>
            <FormGroup>
              <Button block color="primary" type="submit" onClick={login}>Sign in</Button>
            </FormGroup>
          </Form>
        </div>
      </ModalBody>
    </Modal>
  );
}

export default SignInModal;