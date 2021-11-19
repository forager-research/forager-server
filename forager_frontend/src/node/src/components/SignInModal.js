import React, { useContext, useState } from "react";
import {
  Button,
  Form,
  FormGroup,
  Input,
  Modal,
  ModalBody
} from "reactstrap";
import { UserContext } from "../UserContext"

const SignInModal = ({
  isOpen,
  toggle,
}) => {
  const { username, setUsername } = useContext(UserContext);
  const [ loginUsername, setLoginUsername ] = useState("");

  const login = () => {
    if (loginUsername !== null && loginUsername !== undefined && loginUsername.length > 0) {
      setUsername(loginUsername);
      toggle();
    }
  }

  return (
    <Modal
      isOpen={isOpen}
      toggle={toggle}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
      backdrop="static"
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
              <Button block color="primary" type="button" onClick={login}>Sign in</Button>
            </FormGroup>
          </Form>
        </div>
      </ModalBody>
    </Modal>
  );
}

export default SignInModal;


